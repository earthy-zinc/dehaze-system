import { PackageAPI, type SalesStatsVO } from "dehaze-sdk-js";
import { Card, Col, Drawer, Empty, Row, Spin, Statistic } from "antd";
import * as echarts from "echarts";
import React, {
  forwardRef,
  useCallback,
  useImperativeHandle,
  useRef,
  useState,
} from "react";

const LEVEL_COLOR_MAP: Record<string, string> = {
  level_1: "#409eff",
  level_2: "#722ed1",
  level_3: "#fa8c16",
};

export interface SalesStatsDrawerRef {
  open: () => void;
}

const SalesStatsDrawer = forwardRef<SalesStatsDrawerRef>((_props, ref) => {
  const [visible, setVisible] = useState(false);
  const [loading, setLoading] = useState(false);
  const [statsData, setStatsData] = useState<SalesStatsVO | null>(null);

  const packageChartRef = useRef<echarts.ECharts | null>(null);
  const levelChartRef = useRef<echarts.ECharts | null>(null);
  const periodChartRef = useRef<echarts.ECharts | null>(null);
  const couponChartRef = useRef<echarts.ECharts | null>(null);

  const packageElRef = useRef<HTMLDivElement | null>(null);
  const levelElRef = useRef<HTMLDivElement | null>(null);
  const periodElRef = useRef<HTMLDivElement | null>(null);
  const couponElRef = useRef<HTMLDivElement | null>(null);

  const disposeCharts = useCallback(() => {
    packageChartRef.current?.dispose();
    levelChartRef.current?.dispose();
    periodChartRef.current?.dispose();
    couponChartRef.current?.dispose();
    packageChartRef.current = null;
    levelChartRef.current = null;
    periodChartRef.current = null;
    couponChartRef.current = null;
  }, []);

  const initCharts = useCallback((data: SalesStatsVO) => {
    if (packageElRef.current) {
      packageChartRef.current = echarts.init(packageElRef.current);
      packageChartRef.current.setOption({
        tooltip: { trigger: "axis", axisPointer: { type: "cross" } },
        legend: { data: ["销售额", "销量"] },
        grid: { left: "3%", right: "4%", bottom: "10%", containLabel: true },
        xAxis: {
          type: "category",
          data: data.packageStats.map((p) => p.packageName),
          axisPointer: { type: "shadow" },
        },
        yAxis: [
          {
            type: "value",
            name: "销售额",
            axisLabel: { formatter: "¥{value}" },
          },
          { type: "value", name: "销量", axisLabel: { formatter: "{value}" } },
        ],
        series: [
          {
            name: "销售额",
            type: "bar",
            data: data.packageStats.map((p) => p.revenue),
            barWidth: 20,
            itemStyle: {
              color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                { offset: 0, color: "#83bff6" },
                { offset: 1, color: "#188df0" },
              ]),
            },
          },
          {
            name: "销量",
            type: "line",
            yAxisIndex: 1,
            data: data.packageStats.map((p) => p.salesCount),
            itemStyle: { color: "#67C23A" },
          },
        ],
      });
    }

    if (levelElRef.current) {
      levelChartRef.current = echarts.init(levelElRef.current);
      levelChartRef.current.setOption({
        tooltip: {
          trigger: "item",
          formatter: "{a} <br/>{b}: ¥{c} ({d}%)",
        },
        legend: { orient: "vertical", left: "left" },
        series: [
          {
            name: "等级销售额",
            type: "pie",
            radius: "50%",
            data: data.levelStats.map((l) => ({
              name: l.levelName,
              value: l.revenue,
              itemStyle: { color: LEVEL_COLOR_MAP[l.levelCode] },
            })),
            emphasis: {
              itemStyle: {
                shadowBlur: 10,
                shadowOffsetX: 0,
                shadowColor: "rgba(0, 0, 0, 0.5)",
              },
            },
          },
        ],
      });
    }

    if (periodElRef.current) {
      periodChartRef.current = echarts.init(periodElRef.current);
      periodChartRef.current.setOption({
        tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
        grid: { left: "3%", right: "4%", bottom: "10%", containLabel: true },
        xAxis: {
          type: "category",
          data: data.periodStats.map((p) => p.periodName),
          axisPointer: { type: "shadow" },
        },
        yAxis: { type: "value", name: "销量" },
        series: [
          {
            name: "销量",
            type: "bar",
            data: data.periodStats.map((p) => p.salesCount),
            barWidth: 30,
            itemStyle: {
              color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                { offset: 0, color: "#83bff6" },
                { offset: 1, color: "#188df0" },
              ]),
            },
          },
        ],
      });
    }

    if (couponElRef.current) {
      couponChartRef.current = echarts.init(couponElRef.current);
      const rate = Math.round((data.couponStats.usageRate ?? 0) * 100);
      couponChartRef.current.setOption({
        series: [
          {
            name: "使用率",
            type: "gauge",
            startAngle: 90,
            endAngle: -270,
            radius: "85%",
            pointer: { show: false },
            progress: {
              show: true,
              overlap: false,
              roundCap: true,
              clip: false,
              itemStyle: {
                color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
                  { offset: 0, color: "#83bff6" },
                  { offset: 1, color: "#188df0" },
                ]),
              },
            },
            axisLine: {
              lineStyle: {
                width: 20,
                color: [[1, "#e6ebf5"]],
              },
            },
            splitLine: { show: false },
            axisTick: { show: false },
            axisLabel: { show: false },
            data: [{ value: rate, name: "使用率" }],
            title: {
              show: true,
              offsetCenter: [0, "30%"],
              fontSize: 13,
              color: "#666",
            },
            detail: {
              valueAnimation: true,
              formatter: "{value}%",
              fontSize: 22,
              offsetCenter: [0, 0],
              color: "#188df0",
            },
          },
        ],
      });
    }
  }, []);

  const handleResize = useCallback(() => {
    packageChartRef.current?.resize();
    levelChartRef.current?.resize();
    periodChartRef.current?.resize();
    couponChartRef.current?.resize();
  }, []);

  const open = useCallback(() => {
    setVisible(true);
    setLoading(true);
    setStatsData(null);
    PackageAPI.getSalesStats()
      .then((data) => {
        setStatsData(data);
        setTimeout(() => initCharts(data), 50);
      })
      .catch(() => {
        setStatsData(null);
      })
      .finally(() => {
        setLoading(false);
      });
  }, [initCharts]);

  useImperativeHandle(ref, () => ({ open }), [open]);

  const handleClose = useCallback(() => {
    setVisible(false);
    disposeCharts();
  }, [disposeCharts]);

  React.useEffect(() => {
    if (!visible) return;
    window.addEventListener("resize", handleResize);
    return () => {
      window.removeEventListener("resize", handleResize);
    };
  }, [visible, handleResize]);

  React.useEffect(() => {
    return () => {
      disposeCharts();
    };
  }, [disposeCharts]);

  return (
    <Drawer
      title="销售统计"
      open={visible}
      onClose={handleClose}
      width={640}
      destroyOnHidden
    >
      <Spin spinning={loading}>
        {statsData ? (
          <>
            <Row gutter={16} style={{ marginBottom: 16 }}>
              <Col span={12}>
                <Card size="small">
                  <Statistic
                    title="总销售额"
                    value={statsData.totalRevenue}
                    precision={2}
                    prefix="¥"
                  />
                </Card>
              </Col>
              <Col span={12}>
                <Card size="small">
                  <Statistic title="总销量" value={statsData.totalSales} />
                </Card>
              </Col>
            </Row>

            <h4 className="stats-title">各套餐销售统计</h4>
            <div ref={packageElRef} style={{ width: "100%", height: 240 }} />

            <h4 className="stats-title">各等级统计</h4>
            <div ref={levelElRef} style={{ width: "100%", height: 240 }} />

            <h4 className="stats-title">各周期统计</h4>
            <div ref={periodElRef} style={{ width: "100%", height: 240 }} />

            <h4 className="stats-title">优惠券使用统计</h4>
            <Card size="small" style={{ marginBottom: 12 }}>
              <Row>
                <Col span={8}>
                  <Statistic
                    title="累计发放"
                    value={statsData.couponStats.totalIssued}
                  />
                </Col>
                <Col span={8}>
                  <Statistic
                    title="累计使用"
                    value={statsData.couponStats.totalUsed}
                  />
                </Col>
                <Col span={8}>
                  <Statistic
                    title="使用率"
                    value={(statsData.couponStats.usageRate * 100).toFixed(1)}
                    suffix="%"
                  />
                </Col>
              </Row>
            </Card>
            <div ref={couponElRef} style={{ width: "100%", height: 240 }} />
          </>
        ) : (
          !loading && <Empty description="暂无统计数据" />
        )}
      </Spin>
    </Drawer>
  );
});

SalesStatsDrawer.displayName = "SalesStatsDrawer";

export default SalesStatsDrawer;
