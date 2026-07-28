import {
  OrderAPI,
  type OrderStatsVO,
  type OrderStatus,
  type PayMethod,
} from "dehaze-sdk-js";
import { Card, Col, Drawer, Empty, Row, Spin } from "antd";
import * as echarts from "echarts";
import React, {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";

const STATUS_LABEL: Record<OrderStatus, string> = {
  pending: "待支付",
  paid: "已支付",
  completed: "已完成",
  cancelled: "已取消",
  refunding: "退款中",
  refunded: "已退款",
};

const STATUS_COLOR: Record<OrderStatus, string> = {
  pending: "#faad14",
  paid: "#1677ff",
  completed: "#8c8c8c",
  cancelled: "#8c8c8c",
  refunding: "#faad14",
  refunded: "#8c8c8c",
};

const PAY_METHOD_LABEL: Record<PayMethod, string> = {
  wechat: "微信支付",
  alipay: "支付宝",
  balance: "余额支付",
  combined: "组合支付",
};

export interface OrderStatsDrawerRef {
  open: () => void;
}

const OrderStatsDrawer = forwardRef<OrderStatsDrawerRef>((_, ref) => {
  const [visible, setVisible] = useState(false);
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<OrderStatsVO | null>(null);

  const statusChartRef = useRef<echarts.ECharts | null>(null);
  const payMethodChartRef = useRef<echarts.ECharts | null>(null);
  const packageChartRef = useRef<echarts.ECharts | null>(null);
  const dailyChartRef = useRef<echarts.ECharts | null>(null);

  const statusElRef = useRef<HTMLDivElement | null>(null);
  const payMethodElRef = useRef<HTMLDivElement | null>(null);
  const packageElRef = useRef<HTMLDivElement | null>(null);
  const dailyElRef = useRef<HTMLDivElement | null>(null);

  const disposeCharts = useCallback(() => {
    statusChartRef.current?.dispose();
    statusChartRef.current = null;
    payMethodChartRef.current?.dispose();
    payMethodChartRef.current = null;
    packageChartRef.current?.dispose();
    packageChartRef.current = null;
    dailyChartRef.current?.dispose();
    dailyChartRef.current = null;
  }, []);

  const initCharts = useCallback(
    (stats: OrderStatsVO) => {
      disposeCharts();

      if (statusElRef.current) {
        statusChartRef.current = echarts.init(statusElRef.current);
        statusChartRef.current.setOption({
          tooltip: { trigger: "item", formatter: "{a} <br/>{b}: {c} ({d}%)" },
          legend: { bottom: 0 },
          series: [
            {
              name: "订单数",
              type: "pie",
              radius: ["40%", "70%"],
              data: Object.entries(stats.statusDistribution).map(
                ([status, value]) => ({
                  name: STATUS_LABEL[status as OrderStatus],
                  value,
                  itemStyle: { color: STATUS_COLOR[status as OrderStatus] },
                })
              ),
            },
          ],
        });
      }

      if (payMethodElRef.current) {
        payMethodChartRef.current = echarts.init(payMethodElRef.current);
        payMethodChartRef.current.setOption({
          tooltip: { trigger: "item", formatter: "{a} <br/>{b}: {c} ({d}%)" },
          legend: { bottom: 0 },
          series: [
            {
              name: "订单数",
              type: "pie",
              radius: ["40%", "70%"],
              data: Object.entries(stats.payMethodDistribution).map(
                ([method, value]) => ({
                  name: PAY_METHOD_LABEL[method as PayMethod],
                  value,
                })
              ),
            },
          ],
        });
      }

      if (packageElRef.current) {
        packageChartRef.current = echarts.init(packageElRef.current);
        packageChartRef.current.setOption({
          tooltip: { trigger: "axis", axisPointer: { type: "cross" } },
          legend: { bottom: 0, data: ["收入", "订单数"] },
          grid: { left: "2%", right: "5%", bottom: "15%", containLabel: true },
          xAxis: {
            type: "category",
            data: stats.packageDistribution.map((p) => p.packageName),
          },
          yAxis: [
            {
              type: "value",
              name: "收入",
              axisLabel: { formatter: "¥{value}" },
            },
            { type: "value", name: "订单数" },
          ],
          series: [
            {
              name: "收入",
              type: "bar",
              data: stats.packageDistribution.map((p) => p.revenue),
            },
            {
              name: "订单数",
              type: "line",
              yAxisIndex: 1,
              data: stats.packageDistribution.map((p) => p.count),
            },
          ],
        });
      }

      if (dailyElRef.current) {
        dailyChartRef.current = echarts.init(dailyElRef.current);
        dailyChartRef.current.setOption({
          tooltip: { trigger: "axis", axisPointer: { type: "cross" } },
          legend: { bottom: 0, data: ["收入", "订单数"] },
          grid: { left: "2%", right: "5%", bottom: "15%", containLabel: true },
          xAxis: {
            type: "category",
            data: stats.dailyStats.map((d) => d.date),
          },
          yAxis: [
            {
              type: "value",
              name: "收入",
              axisLabel: { formatter: "¥{value}" },
            },
            { type: "value", name: "订单数" },
          ],
          series: [
            {
              name: "收入",
              type: "line",
              yAxisIndex: 0,
              data: stats.dailyStats.map((d) => d.revenue),
            },
            {
              name: "订单数",
              type: "bar",
              yAxisIndex: 1,
              data: stats.dailyStats.map((d) => d.count),
            },
          ],
        });
      }
    },
    [disposeCharts]
  );

  const handleAfterOpenChange = useCallback(
    (open: boolean) => {
      if (open) {
        setLoading(true);
        setData(null);
        OrderAPI.getStats()
          .then((stats) => {
            setData(stats);
          })
          .finally(() => {
            setLoading(false);
          });
      } else {
        disposeCharts();
      }
    },
    [disposeCharts]
  );

  const open = useCallback(() => {
    setVisible(true);
  }, []);

  useImperativeHandle(ref, () => ({ open }), [open]);

  useEffect(() => {
    if (!data || !visible) return;
    const timer = setTimeout(() => initCharts(data), 0);
    return () => clearTimeout(timer);
  }, [data, visible, initCharts]);

  useEffect(() => {
    const handleResize = () => {
      statusChartRef.current?.resize();
      payMethodChartRef.current?.resize();
      packageChartRef.current?.resize();
      dailyChartRef.current?.resize();
    };
    if (visible) {
      window.addEventListener("resize", handleResize);
    }
    return () => {
      window.removeEventListener("resize", handleResize);
    };
  }, [visible]);

  useEffect(() => () => disposeCharts(), [disposeCharts]);

  return (
    <Drawer
      title="订单统计"
      open={visible}
      width={560}
      destroyOnHidden
      afterOpenChange={handleAfterOpenChange}
    >
      <Spin spinning={loading}>
        {data ? (
          <>
            <Row gutter={12}>
              <Col span={12}>
                <Card size="small">
                  <div className="stats-label">总订单数</div>
                  <div className="stats-value">{data.totalOrders}</div>
                </Card>
              </Col>
              <Col span={12}>
                <Card size="small">
                  <div className="stats-label">总收入</div>
                  <div className="stats-value">
                    ¥{data.totalRevenue.toFixed(2)}
                  </div>
                </Card>
              </Col>
              <Col span={12}>
                <Card size="small">
                  <div className="stats-label">总退款</div>
                  <div className="stats-value">
                    ¥{data.totalRefund.toFixed(2)}
                  </div>
                </Card>
              </Col>
              <Col span={12}>
                <Card size="small">
                  <div className="stats-label">退款率</div>
                  <div className="stats-value">
                    {(data.refundRate * 100).toFixed(2)}%
                  </div>
                </Card>
              </Col>
            </Row>

            <div className="stats-section-title">状态分布</div>
            <div ref={statusElRef} style={{ width: "100%", height: 260 }} />

            <div className="stats-section-title">支付方式分布</div>
            <div ref={payMethodElRef} style={{ width: "100%", height: 260 }} />

            <div className="stats-section-title">套餐分布</div>
            <div ref={packageElRef} style={{ width: "100%", height: 260 }} />

            <div className="stats-section-title">每日趋势</div>
            <div ref={dailyElRef} style={{ width: "100%", height: 260 }} />
          </>
        ) : (
          !loading && <Empty description="暂无数据" />
        )}
      </Spin>
    </Drawer>
  );
});

OrderStatsDrawer.displayName = "OrderStatsDrawer";

export default OrderStatsDrawer;
