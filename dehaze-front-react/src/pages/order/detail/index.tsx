import {
  OrderAPI,
  type OrderDetailVO,
  type OrderStatus,
  type PayMethod,
  type PaymentRecordVO,
  type RefundStatus,
} from "dehaze-sdk-js";
import {
  Button,
  Card,
  Divider,
  Empty,
  Input,
  Modal,
  Spin,
  Table,
  Tag,
  type TableColumnsType,
  message,
} from "antd";
import {
  ArrowLeftOutlined,
  ClockCircleOutlined,
  FileTextOutlined,
  WalletOutlined,
} from "@ant-design/icons";
import React, { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import PaymentDialog, {
  type PaymentDialogRef,
} from "./components/PaymentDialog";
import "./index.scss";

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

const OrderDetail: React.FC = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const orderNo = searchParams.get("orderNo") || "";

  const [loading, setLoading] = useState(false);
  const [order, setOrder] = useState<OrderDetailVO | null>(null);

  const [cancelModal, setCancelModal] = useState({
    visible: false,
    reason: "",
  });

  const paymentDialogRef = useRef<PaymentDialogRef>(null);

  const loadDetail = useCallback(async (no: string) => {
    if (!no) {
      setOrder(null);
      return;
    }
    setLoading(true);
    try {
      const data = await OrderAPI.getDetail(no);
      setOrder(data);
    } catch {
      setOrder(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadDetail(orderNo);
  }, [orderNo, loadDetail]);

  const goBack = useCallback(() => {
    navigate("/order/my");
  }, [navigate]);

  const handleOpenPayment = useCallback(() => {
    if (!order) return;
    paymentDialogRef.current?.open(order.orderNo, order.payableAmount);
  }, [order]);

  const handleCancelClick = useCallback(() => {
    setCancelModal({ visible: true, reason: "" });
  }, []);

  const handleCancelSubmit = useCallback(async () => {
    if (!order) return;
    if (!cancelModal.reason.trim()) {
      message.warning("取消原因不能为空");
      return;
    }
    try {
      await OrderAPI.cancel(order.orderNo, cancelModal.reason.trim());
      message.success("订单已取消");
      setCancelModal({ visible: false, reason: "" });
      loadDetail(order.orderNo);
    } catch (error: any) {
      message.error(error?.message || "取消失败");
    }
  }, [order, cancelModal.reason, loadDetail]);

  const handleApplyRefund = useCallback(() => {
    navigate("/order/my");
  }, [navigate]);

  const handlePaidSuccess = useCallback(() => {
    loadDetail(orderNo);
  }, [orderNo, loadDetail]);

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
  ];

  const showFooterActions = order
    ? order.status === "pending" || order.status === "paid"
    : false;

  return (
    <div className="order-detail-page">
      <Spin spinning={loading}>
        {order ? (
          <div className="detail-wrapper">
            <div className="detail-header">
              <Button type="link" icon={<ArrowLeftOutlined />} onClick={goBack}>
                返回我的订单
              </Button>
            </div>

            <div className={`status-card status-${order.status}`}>
              <div className="status-card-stripe" />
              <div className="status-card-content">
                <div className="status-row">
                  <span className="status-text">
                    {STATUS_MAP[order.status].label}
                  </span>
                  <Tag color={STATUS_MAP[order.status].color}>
                    {STATUS_MAP[order.status].label}
                  </Tag>
                </div>
                <div className="meta-row">
                  <span className="order-no">订单号：{order.orderNo}</span>
                  <span className="create-time">
                    创建时间：{order.createTime}
                  </span>
                </div>
                {order.cancelReason && (
                  <div className="cancel-reason">
                    取消原因：{order.cancelReason}
                  </div>
                )}
              </div>
            </div>

            <Card className="info-card" size="small" bordered={false}>
              <div className="info-card-title">套餐信息</div>
              <div className="info-card-body">
                <div className="package-name">
                  <FileTextOutlined />
                  <span>{order.packageName}</span>
                  <Tag color="default">{order.packageLevel}</Tag>
                </div>
                {(order.effectiveTime || order.expireTime) && (
                  <div className="validity">
                    <ClockCircleOutlined />
                    {order.effectiveTime && (
                      <span>生效：{order.effectiveTime}</span>
                    )}
                    {order.expireTime && <span>到期：{order.expireTime}</span>}
                  </div>
                )}
              </div>
            </Card>

            <Card className="info-card" size="small" bordered={false}>
              <div className="info-card-title">金额明细</div>
              <div className="info-card-body">
                <div className="amount-row">
                  <span className="amount-label">订单原价</span>
                  <span className="amount-value">
                    ¥{order.originalPrice.toFixed(2)}
                  </span>
                </div>
                {order.discountAmount > 0 && (
                  <div className="amount-row discount">
                    <span className="amount-label">折扣优惠</span>
                    <span className="amount-value">
                      -¥{order.discountAmount.toFixed(2)}
                    </span>
                  </div>
                )}
                {order.couponAmount > 0 && (
                  <div className="amount-row discount">
                    <span className="amount-label">优惠券抵扣</span>
                    <span className="amount-value">
                      -¥{order.couponAmount.toFixed(2)}
                    </span>
                  </div>
                )}
                <Divider style={{ margin: "8px 0" }} />
                <div className="amount-row total">
                  <span className="amount-label">实付金额</span>
                  <span className="amount-value payable">
                    ¥{order.payableAmount.toFixed(2)}
                  </span>
                </div>
                {order.paidAmount > 0 && (
                  <div className="amount-row">
                    <span className="amount-label">已支付金额</span>
                    <span className="amount-value">
                      ¥{order.paidAmount.toFixed(2)}
                    </span>
                  </div>
                )}
              </div>
            </Card>

            <Card className="info-card" size="small" bordered={false}>
              <div className="info-card-title">支付信息</div>
              <div className="info-card-body">
                <div className="amount-row">
                  <span className="amount-label">支付方式</span>
                  <span className="amount-value">
                    {order.payMethod ? PAY_METHOD_LABEL[order.payMethod] : "-"}
                  </span>
                </div>
                <div className="amount-row">
                  <span className="amount-label">支付时间</span>
                  <span className="amount-value">{order.paidTime || "-"}</span>
                </div>
                {order.paymentRecords && order.paymentRecords.length > 0 && (
                  <div className="payment-records">
                    <div className="payment-records-title">支付流水</div>
                    <Table
                      size="small"
                      rowKey="id"
                      pagination={false}
                      columns={paymentColumns}
                      dataSource={order.paymentRecords}
                      style={{ marginTop: 8 }}
                    />
                  </div>
                )}
              </div>
            </Card>

            {order.refundRecord && (
              <Card
                className="info-card refund-card"
                size="small"
                bordered={false}
              >
                <div className="info-card-title">退款信息</div>
                <div className="info-card-body">
                  <div className="amount-row">
                    <span className="amount-label">退款单号</span>
                    <span className="amount-value">
                      {order.refundRecord.refundNo}
                    </span>
                  </div>
                  <div className="amount-row">
                    <span className="amount-label">退款金额</span>
                    <span className="amount-value refund-amount">
                      ¥{order.refundRecord.refundAmount.toFixed(2)}
                    </span>
                  </div>
                  <div className="amount-row">
                    <span className="amount-label">退款原因</span>
                    <span className="amount-value">
                      {order.refundRecord.reason}
                    </span>
                  </div>
                  <div className="amount-row">
                    <span className="amount-label">已用配额</span>
                    <span className="amount-value">
                      {order.refundRecord.usedQuota}
                    </span>
                  </div>
                  <div className="amount-row">
                    <span className="amount-label">退款状态</span>
                    <Tag
                      color={REFUND_STATUS_MAP[order.refundRecord.status].color}
                    >
                      {REFUND_STATUS_MAP[order.refundRecord.status].label}
                    </Tag>
                  </div>
                  <div className="amount-row">
                    <span className="amount-label">申请时间</span>
                    <span className="amount-value">
                      {order.refundRecord.applyTime}
                    </span>
                  </div>
                  {order.refundRecord.auditTime && (
                    <div className="amount-row">
                      <span className="amount-label">审核时间</span>
                      <span className="amount-value">
                        {order.refundRecord.auditTime}
                      </span>
                    </div>
                  )}
                  {order.refundRecord.refundTime && (
                    <div className="amount-row">
                      <span className="amount-label">退款时间</span>
                      <span className="amount-value">
                        {order.refundRecord.refundTime}
                      </span>
                    </div>
                  )}
                  {order.refundRecord.auditRemark && (
                    <div className="amount-row">
                      <span className="amount-label">审核备注</span>
                      <span className="amount-value">
                        {order.refundRecord.auditRemark}
                      </span>
                    </div>
                  )}
                  {order.refundRecord.errorMessage && (
                    <div className="amount-row">
                      <span className="amount-label">错误信息</span>
                      <span className="amount-value error-text">
                        {order.refundRecord.errorMessage}
                      </span>
                    </div>
                  )}
                </div>
              </Card>
            )}

            {showFooterActions && (
              <div className="footer-action-bar">
                {order.status === "pending" && (
                  <>
                    <Button type="primary" onClick={handleOpenPayment}>
                      立即支付
                    </Button>
                    <Button onClick={handleCancelClick}>取消订单</Button>
                  </>
                )}
                {order.status === "paid" && (
                  <Button icon={<WalletOutlined />} onClick={handleApplyRefund}>
                    申请退款
                  </Button>
                )}
              </div>
            )}
          </div>
        ) : (
          !loading && (
            <Empty description="订单不存在或已被删除">
              <Button type="primary" onClick={goBack}>
                返回我的订单
              </Button>
            </Empty>
          )
        )}
      </Spin>

      <Modal
        title="取消订单"
        open={cancelModal.visible}
        onOk={handleCancelSubmit}
        onCancel={() => setCancelModal({ visible: false, reason: "" })}
        okText="确定取消"
        cancelText="返回"
      >
        <Input.TextArea
          rows={3}
          value={cancelModal.reason}
          placeholder="请输入取消原因"
          onChange={(e) =>
            setCancelModal({ visible: true, reason: e.target.value })
          }
        />
      </Modal>

      <PaymentDialog ref={paymentDialogRef} onPaid={handlePaidSuccess} />
    </div>
  );
};

export default OrderDetail;
