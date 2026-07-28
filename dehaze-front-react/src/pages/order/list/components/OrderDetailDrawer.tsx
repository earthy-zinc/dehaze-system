import {
  OrderAPI,
  type OrderDetailVO,
  type OrderStatus,
  type PayMethod,
  type PaymentRecordVO,
  type RefundStatus,
} from "dehaze-sdk-js";
import {
  Descriptions,
  Drawer,
  Empty,
  Spin,
  Table,
  Tabs,
  Tag,
  type TableColumnsType,
} from "antd";
import React, {
  forwardRef,
  useCallback,
  useImperativeHandle,
  useState,
} from "react";

const STATUS_MAP: Record<OrderStatus, { label: string; color: string }> = {
  pending: { label: "待支付", color: "warning" },
  paid: { label: "已支付", color: "processing" },
  completed: { label: "已完成", color: "default" },
  cancelled: { label: "已取消", color: "default" },
  refunding: { label: "退款中", color: "warning" },
  refunded: { label: "已退款", color: "default" },
};

const PAY_METHOD_LABEL: Record<PayMethod, string> = {
  wechat: "微信支付",
  alipay: "支付宝",
  balance: "余额支付",
  combined: "组合支付",
};

const REFUND_STATUS_MAP: Record<
  RefundStatus,
  { label: string; color: string }
> = {
  refunding: { label: "退款中", color: "warning" },
  refunded: { label: "退款成功", color: "default" },
  refund_failed: { label: "退款失败", color: "error" },
};

export interface OrderDetailDrawerRef {
  open: (orderNo: string) => void;
}

const OrderDetailDrawer = forwardRef<OrderDetailDrawerRef>((_, ref) => {
  const [visible, setVisible] = useState(false);
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<OrderDetailVO | null>(null);

  const open = useCallback(async (orderNo: string) => {
    setVisible(true);
    setLoading(true);
    setData(null);
    try {
      const detail = await OrderAPI.getDetail(orderNo);
      setData(detail);
    } catch {
      setData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useImperativeHandle(ref, () => ({ open }), [open]);

  const handleClose = useCallback(() => {
    setVisible(false);
    setData(null);
  }, []);

  const paymentColumns: TableColumnsType<PaymentRecordVO> = [
    { title: "流水号", dataIndex: "paymentNo", key: "paymentNo" },
    {
      title: "渠道",
      dataIndex: "channel",
      key: "channel",
      width: 120,
      render: (channel: PayMethod) => PAY_METHOD_LABEL[channel],
    },
    {
      title: "金额",
      dataIndex: "amount",
      key: "amount",
      width: 100,
      align: "right",
      render: (amount: number) => `¥${amount.toFixed(2)}`,
    },
    {
      title: "状态",
      dataIndex: "status",
      key: "status",
      width: 80,
      align: "center",
    },
    {
      title: "回调时间",
      dataIndex: "callbackTime",
      key: "callbackTime",
      width: 170,
      render: (v?: string) => v || "-",
    },
    {
      title: "创建时间",
      dataIndex: "createTime",
      key: "createTime",
      width: 170,
    },
  ];

  return (
    <Drawer
      title="订单详情"
      open={visible}
      width={760}
      destroyOnHidden
      onClose={handleClose}
    >
      <Spin spinning={loading}>
        {data && (
          <>
            <Descriptions column={2} bordered size="small">
              <Descriptions.Item label="订单号">
                {data.orderNo}
              </Descriptions.Item>
              <Descriptions.Item label="状态">
                <Tag color={STATUS_MAP[data.status].color}>
                  {STATUS_MAP[data.status].label}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="用户">
                {data.username}
              </Descriptions.Item>
              <Descriptions.Item label="套餐">
                {data.packageName} ({data.packageLevel})
              </Descriptions.Item>
              <Descriptions.Item label="原价">
                ¥{data.originalPrice.toFixed(2)}
              </Descriptions.Item>
              <Descriptions.Item label="折扣优惠">
                ¥{data.discountAmount.toFixed(2)}
              </Descriptions.Item>
              <Descriptions.Item label="优惠券抵扣">
                ¥{data.couponAmount.toFixed(2)}
              </Descriptions.Item>
              <Descriptions.Item label="实付">
                ¥{data.payableAmount.toFixed(2)}
              </Descriptions.Item>
              <Descriptions.Item label="支付方式">
                {data.payMethod ? PAY_METHOD_LABEL[data.payMethod] : "-"}
              </Descriptions.Item>
              <Descriptions.Item label="已支付金额">
                ¥{data.paidAmount.toFixed(2)}
              </Descriptions.Item>
              <Descriptions.Item label="创建时间">
                {data.createTime}
              </Descriptions.Item>
              <Descriptions.Item label="支付时间">
                {data.paidTime || "-"}
              </Descriptions.Item>
              <Descriptions.Item label="生效时间">
                {data.effectiveTime || "-"}
              </Descriptions.Item>
              <Descriptions.Item label="到期时间">
                {data.expireTime}
              </Descriptions.Item>
              <Descriptions.Item label="自动续费">
                {data.isAutoRenew ? "是" : "否"}
              </Descriptions.Item>
              {data.cancelReason && (
                <Descriptions.Item label="取消原因">
                  {data.cancelReason}
                </Descriptions.Item>
              )}
            </Descriptions>

            <Tabs
              defaultActiveKey="payment"
              style={{ marginTop: 16 }}
              items={[
                {
                  key: "payment",
                  label: "支付流水",
                  children:
                    data.paymentRecords && data.paymentRecords.length > 0 ? (
                      <Table
                        size="small"
                        rowKey="id"
                        pagination={false}
                        columns={paymentColumns}
                        dataSource={data.paymentRecords}
                      />
                    ) : (
                      <Empty
                        description="无支付流水"
                        image={Empty.PRESENTED_IMAGE_SIMPLE}
                      />
                    ),
                },
                {
                  key: "refund",
                  label: "退款信息",
                  children: data.refundRecord ? (
                    <Descriptions column={2} bordered size="small">
                      <Descriptions.Item label="退款单号">
                        {data.refundRecord.refundNo}
                      </Descriptions.Item>
                      <Descriptions.Item label="退款金额">
                        ¥{data.refundRecord.refundAmount.toFixed(2)}
                      </Descriptions.Item>
                      <Descriptions.Item label="退款原因">
                        {data.refundRecord.reason}
                      </Descriptions.Item>
                      <Descriptions.Item label="已用配额">
                        {data.refundRecord.usedQuota}
                      </Descriptions.Item>
                      <Descriptions.Item label="状态">
                        <Tag
                          color={
                            REFUND_STATUS_MAP[data.refundRecord.status].color
                          }
                        >
                          {REFUND_STATUS_MAP[data.refundRecord.status].label}
                        </Tag>
                      </Descriptions.Item>
                      <Descriptions.Item label="申请时间">
                        {data.refundRecord.applyTime}
                      </Descriptions.Item>
                      <Descriptions.Item label="审核时间">
                        {data.refundRecord.auditTime || "-"}
                      </Descriptions.Item>
                      <Descriptions.Item label="退款时间">
                        {data.refundRecord.refundTime || "-"}
                      </Descriptions.Item>
                      {data.refundRecord.auditRemark && (
                        <Descriptions.Item label="审核备注">
                          {data.refundRecord.auditRemark}
                        </Descriptions.Item>
                      )}
                      {data.refundRecord.errorMessage && (
                        <Descriptions.Item label="错误信息">
                          {data.refundRecord.errorMessage}
                        </Descriptions.Item>
                      )}
                    </Descriptions>
                  ) : (
                    <Empty
                      description="无退款记录"
                      image={Empty.PRESENTED_IMAGE_SIMPLE}
                    />
                  ),
                },
              ]}
            />
          </>
        )}
      </Spin>
    </Drawer>
  );
});

OrderDetailDrawer.displayName = "OrderDetailDrawer";

export default OrderDetailDrawer;
