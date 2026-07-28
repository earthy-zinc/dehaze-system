import {
  OrderAPI,
  type MyOrderQuery,
  type MyOrderVO,
  type OrderStatus,
  type PayMethod,
  type RefundApplyForm,
} from "dehaze-sdk-js";
import {
  Alert,
  Button,
  Empty,
  Form,
  Input,
  Modal,
  Pagination,
  Select,
  Space,
  Spin,
  Tag,
  message,
} from "antd";
import { ClockCircleOutlined, WalletOutlined } from "@ant-design/icons";
import React, { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import "./index.scss";

const STATUS_TABS: { label: string; value: OrderStatus | undefined }[] = [
  { label: "全部", value: undefined },
  { label: "待支付", value: "pending" },
  { label: "已支付", value: "paid" },
  { label: "已完成", value: "completed" },
  { label: "已取消", value: "cancelled" },
  { label: "退款中", value: "refunding" },
  { label: "已退款", value: "refunded" },
];

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

const REFUND_REASON_OPTIONS = [
  "功能不满足需求",
  "使用体验不佳",
  "重复购买",
  "暂不需要",
  "其他原因",
];

interface RefundFormValues extends RefundApplyForm {
  reason: string;
  customReason?: string;
}

const MyOrders: React.FC = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [orderList, setOrderList] = useState<MyOrderVO[]>([]);
  const [total, setTotal] = useState(0);
  const [activeStatus, setActiveStatus] = useState<OrderStatus | undefined>(
    undefined
  );
  const [queryParams, setQueryParams] = useState<MyOrderQuery>({
    pageNum: 1,
    pageSize: 10,
  });

  const [cancelModal, setCancelModal] = useState<{
    visible: boolean;
    orderNo: string;
    reason: string;
  }>({ visible: false, orderNo: "", reason: "" });

  const [refundModal, setRefundModal] = useState<{
    visible: boolean;
    loading: boolean;
    row: MyOrderVO | null;
  }>({ visible: false, loading: false, row: null });
  const [refundForm] = Form.useForm<RefundFormValues>();

  const loadData = useCallback(async (params: MyOrderQuery) => {
    setLoading(true);
    try {
      const result = await OrderAPI.listMy(params);
      setOrderList(result.list || []);
      setTotal(result.total || 0);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData(queryParams);
  }, [queryParams]);

  const handleTabChange = useCallback((value: OrderStatus | undefined) => {
    setActiveStatus(value);
    setQueryParams((prev) => ({ ...prev, status: value, pageNum: 1 }));
  }, []);

  const handlePageChange = useCallback((page: number, pageSize: number) => {
    setQueryParams((prev) => ({ ...prev, pageNum: page, pageSize }));
  }, []);

  const refreshList = useCallback(() => {
    setQueryParams((prev) => ({ ...prev }));
    loadData(queryParams);
  }, [queryParams, loadData]);

  const goDetail = useCallback(
    (order: MyOrderVO) => {
      navigate(`/order/detail?orderNo=${order.orderNo}`);
    },
    [navigate]
  );

  const handleCancelClick = useCallback((order: MyOrderVO) => {
    setCancelModal({ visible: true, orderNo: order.orderNo, reason: "" });
  }, []);

  const handleCancelSubmit = useCallback(async () => {
    if (!cancelModal.reason.trim()) {
      message.warning("取消原因不能为空");
      return;
    }
    try {
      await OrderAPI.cancel(cancelModal.orderNo, cancelModal.reason.trim());
      message.success("订单已取消");
      setCancelModal({ visible: false, orderNo: "", reason: "" });
      refreshList();
    } catch (error: any) {
      message.error(error?.message || "取消失败");
    }
  }, [cancelModal, refreshList]);

  const openRefundDialog = useCallback(
    (order: MyOrderVO) => {
      setRefundModal({ visible: true, loading: false, row: order });
      refundForm.resetFields();
    },
    [refundForm]
  );

  const closeRefundDialog = useCallback(() => {
    setRefundModal({ visible: false, loading: false, row: null });
    refundForm.resetFields();
  }, [refundForm]);

  const handleRefundSubmit = useCallback(async () => {
    if (!refundModal.row) return;
    try {
      const values = await refundForm.validateFields();
      setRefundModal((prev) => ({ ...prev, loading: true }));
      const payload: RefundApplyForm = {
        reason: values.reason,
        customReason:
          values.reason === "其他原因" ? values.customReason : undefined,
      };
      await OrderAPI.applyRefund(refundModal.row.orderNo, payload);
      message.success("退款申请已提交");
      closeRefundDialog();
      refreshList();
    } catch (error: any) {
      if (error?.errorFields) return;
      message.error(error?.message || "退款申请失败");
    } finally {
      setRefundModal((prev) => ({ ...prev, loading: false }));
    }
  }, [refundModal.row, refundForm, closeRefundDialog, refreshList]);

  const hasDiscount = useCallback((order: MyOrderVO) => {
    return order.paidAmount > order.payableAmount && order.payableAmount > 0;
  }, []);

  const renderActions = useCallback(
    (order: MyOrderVO) => {
      if (order.status === "pending") {
        return (
          <Space size="small">
            <Button type="primary" size="small" onClick={() => goDetail(order)}>
              去支付
            </Button>
            <Button size="small" onClick={() => handleCancelClick(order)}>
              取消订单
            </Button>
          </Space>
        );
      }
      if (order.status === "paid") {
        return (
          <Space size="small">
            <Button size="small" onClick={() => openRefundDialog(order)}>
              申请退款
            </Button>
            <Button size="small" onClick={() => goDetail(order)}>
              查看详情
            </Button>
          </Space>
        );
      }
      return (
        <Button size="small" onClick={() => goDetail(order)}>
          查看详情
        </Button>
      );
    },
    [goDetail, handleCancelClick, openRefundDialog]
  );

  const refundReason = refundForm.getFieldValue("reason");

  return (
    <div className="my-order-center">
      <div className="page-header">
        <span className="title-text">我的订单</span>
      </div>

      <div className="status-tabs">
        {STATUS_TABS.map((tab) => (
          <button
            key={tab.value ?? "all"}
            className={`status-tab ${activeStatus === tab.value ? "active" : ""}`}
            onClick={() => handleTabChange(tab.value)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <Spin spinning={loading}>
        <div className="order-list">
          {orderList.length > 0
            ? orderList.map((order) => (
                <div
                  key={order.id}
                  className={`order-card status-${order.status}`}
                >
                  <div className="card-header">
                    <div className="header-left">
                      <span className="order-no">订单号：{order.orderNo}</span>
                      <span className="create-time">{order.createTime}</span>
                    </div>
                    <Tag color={STATUS_MAP[order.status].color}>
                      {STATUS_MAP[order.status].label}
                    </Tag>
                  </div>

                  <div className="card-body">
                    <div className="package-info">
                      <WalletOutlined className="info-icon" />
                      <div className="info-content">
                        <div className="package-name">
                          {order.packageName}
                          <Tag color="default">{order.packageLevel}</Tag>
                        </div>
                        {order.packageExpireTime && (
                          <div className="package-expire">
                            <ClockCircleOutlined />
                            到期时间：{order.packageExpireTime}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="card-footer">
                    <div className="amount-area">
                      <span className="payable-amount">
                        ¥{order.payableAmount.toFixed(2)}
                      </span>
                      {(order.payableAmount < order.paidAmount ||
                        hasDiscount(order)) && (
                        <span className="original-amount">
                          ¥{order.paidAmount.toFixed(2)}
                        </span>
                      )}
                      {order.payMethod && (
                        <span className="pay-method-text">
                          {PAY_METHOD_LABEL[order.payMethod]}
                        </span>
                      )}
                    </div>
                    <div className="action-area">{renderActions(order)}</div>
                  </div>
                </div>
              ))
            : !loading && (
                <Empty
                  description="暂无订单"
                  image={Empty.PRESENTED_IMAGE_SIMPLE}
                />
              )}
        </div>
      </Spin>

      {total > 0 && (
        <Pagination
          className="order-pagination"
          current={queryParams.pageNum}
          pageSize={queryParams.pageSize}
          total={total}
          showSizeChanger
          showQuickJumper
          pageSizeOptions={["10", "20", "50"]}
          showTotal={(t) => `共 ${t} 条`}
          onChange={handlePageChange}
        />
      )}

      <Modal
        title="取消订单"
        open={cancelModal.visible}
        onOk={handleCancelSubmit}
        onCancel={() =>
          setCancelModal({ visible: false, orderNo: "", reason: "" })
        }
        okText="确定取消"
        cancelText="返回"
      >
        <Input.TextArea
          rows={3}
          value={cancelModal.reason}
          placeholder="请输入取消原因"
          onChange={(e) =>
            setCancelModal((prev) => ({ ...prev, reason: e.target.value }))
          }
        />
      </Modal>

      <Modal
        title="申请退款"
        open={refundModal.visible}
        width={500}
        confirmLoading={refundModal.loading}
        onOk={handleRefundSubmit}
        onCancel={closeRefundDialog}
        okText="提交申请"
        cancelText="取消"
        destroyOnHidden
      >
        {refundModal.row && (
          <>
            <div className="refund-order-info">
              <div className="refund-info-row">
                <span className="label">订单号：</span>
                <span>{refundModal.row.orderNo}</span>
              </div>
              <div className="refund-info-row">
                <span className="label">套餐：</span>
                <span>
                  {refundModal.row.packageName} ({refundModal.row.packageLevel})
                </span>
              </div>
              <div className="refund-info-row">
                <span className="label">退款金额：</span>
                <span className="refund-amount">
                  ¥{refundModal.row.paidAmount.toFixed(2)}
                </span>
              </div>
            </div>

            <Alert
              type="warning"
              showIcon
              message="退款规则"
              description={
                <div className="refund-rules">
                  <p>· 购买 7 天内可申请退款</p>
                  <p>· 已使用配额超过 50% 不可退款</p>
                  <p>· 退款金额按未使用配额比例退还</p>
                  <p>· 退款审核通过后将原路退回</p>
                </div>
              }
              style={{ marginBottom: 16 }}
            />

            <Form form={refundForm} layout="vertical">
              <Form.Item
                name="reason"
                label="退款原因"
                rules={[{ required: true, message: "请选择退款原因" }]}
              >
                <Select
                  placeholder="请选择退款原因"
                  options={REFUND_REASON_OPTIONS.map((r) => ({
                    label: r,
                    value: r,
                  }))}
                />
              </Form.Item>
              {refundReason === "其他原因" && (
                <Form.Item
                  name="customReason"
                  label="具体原因"
                  rules={[{ required: true, message: "请描述具体退款原因" }]}
                >
                  <Input.TextArea rows={3} placeholder="请描述具体退款原因" />
                </Form.Item>
              )}
            </Form>
          </>
        )}
      </Modal>
    </div>
  );
};

export default MyOrders;
