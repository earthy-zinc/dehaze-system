import React, { useState, useEffect, useCallback } from "react";
import { View, Text, ScrollView } from "@tarojs/components";
import Taro from "@tarojs/taro";
import { Tag, Loading, Empty, Tabs, Popup, Input } from "@taroify/core";
import { OrderAPI } from "dehaze-sdk-js";
import type { OrderPageVO, RefundRecordVO, OrderStatsVO } from "dehaze-sdk-js";
import PageLayout from "@/layout";
import { usePermission } from "@/hooks/usePermission";
import { getErrorMessage } from "@/utils/error";
import "./index.less";

const STATUS_LABELS: Record<string, string> = {
  pending: "待支付",
  paid: "已支付",
  completed: "已完成",
  cancelled: "已取消",
  refunding: "退款中",
  refunded: "已退款",
};

const REFUND_STATUS_LABELS: Record<string, string> = {
  refunding: "退款中",
  refunded: "退款成功",
  refund_failed: "退款失败",
};

const OrderManagePage: React.FC = () => {
  const { hasPermission } = usePermission();
  const canViewRefund = hasPermission("sys:order:refund");

  const [tab, setTab] = useState(0);
  const [orders, setOrders] = useState<OrderPageVO[]>([]);
  const [refunds, setRefunds] = useState<RefundRecordVO[]>([]);
  const [stats, setStats] = useState<OrderStatsVO | null>(null);
  const [loading, setLoading] = useState(false);
  const [totalOrders, setTotalOrders] = useState(0);
  const [totalRefunds, setTotalRefunds] = useState(0);
  const [pageNum, setPageNum] = useState(1);
  const [refundPageNum, setRefundPageNum] = useState(1);
  const [keyword, setKeyword] = useState("");
  const [statusFilter, setStatusFilter] = useState("");

  const [auditPopupVisible, setAuditPopupVisible] = useState(false);
  const [auditingRefund, setAuditingRefund] = useState<RefundRecordVO | null>(
    null
  );
  const [auditRemark, setAuditRemark] = useState("");

  const fetchOrders = useCallback(
    async (page: number, kw: string, status: string) => {
      setLoading(true);
      try {
        const params: any = { pageNum: page, pageSize: 15 };
        if (kw) params.keywords = kw;
        if (status) params.status = status;
        const res = await OrderAPI.getPage(params);
        setOrders(res.list);
        setTotalOrders(res.total);
        setPageNum(page);
      } catch (err: unknown) {
        Taro.showToast({
          title: getErrorMessage(err, "加载订单失败"),
          icon: "none",
        });
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const fetchRefunds = useCallback(async (page: number) => {
    try {
      const res = await OrderAPI.listRefunds({ pageNum: page, pageSize: 15 });
      setRefunds(res.list);
      setTotalRefunds(res.total);
      setRefundPageNum(page);
    } catch {
      // 静默
    }
  }, []);

  const fetchStats = useCallback(async () => {
    try {
      const s = await OrderAPI.getStats();
      setStats(s);
    } catch {
      // 静默
    }
  }, []);

  useEffect(() => {
    fetchOrders(1, "", "");
    fetchRefunds(1);
    fetchStats();
  }, [fetchOrders, fetchRefunds, fetchStats]);

  const handleSearch = () => {
    fetchOrders(1, keyword, statusFilter);
  };

  const handleLoadMoreOrders = () => {
    if (orders.length < totalOrders) {
      fetchOrders(pageNum + 1, keyword, statusFilter);
    }
  };

  const handleLoadMoreRefunds = () => {
    if (refunds.length < totalRefunds) {
      fetchRefunds(refundPageNum + 1);
    }
  };

  const handleApproveRefund = async () => {
    if (!auditingRefund) return;
    try {
      await OrderAPI.approveRefund(auditingRefund.id, {
        approved: true,
        remark: auditRemark,
      });
      Taro.showToast({ title: "退款已通过", icon: "success" });
      setAuditPopupVisible(false);
      fetchRefunds(1);
    } catch (err: unknown) {
      Taro.showToast({ title: getErrorMessage(err, "操作失败"), icon: "none" });
    }
  };

  const handleRejectRefund = async () => {
    if (!auditingRefund) return;
    try {
      await OrderAPI.rejectRefund(auditingRefund.id, {
        approved: false,
        remark: auditRemark,
      });
      Taro.showToast({ title: "退款已驳回", icon: "success" });
      setAuditPopupVisible(false);
      fetchRefunds(1);
    } catch (err: unknown) {
      Taro.showToast({ title: getErrorMessage(err, "操作失败"), icon: "none" });
    }
  };

  const openAudit = (refund: RefundRecordVO) => {
    setAuditingRefund(refund);
    setAuditRemark("");
    setAuditPopupVisible(true);
  };

  return (
    <PageLayout level="L2" title="订单管理">
      <View className="system-manage-page">
        {/* 统计概览 */}
        {stats && (
          <View className="stats-bar">
            <View className="stat-item">
              <Text className="stat-value">{stats.totalOrders}</Text>
              <Text className="stat-label">总订单</Text>
            </View>
            <View className="stat-item">
              <Text className="stat-value">{stats.totalRevenue}</Text>
              <Text className="stat-label">总收入</Text>
            </View>
            <View className="stat-item">
              <Text className="stat-value">{stats.refundRate}%</Text>
              <Text className="stat-label">退款率</Text>
            </View>
          </View>
        )}

        <Tabs value={tab} onChange={setTab}>
          <Tabs.TabPane title="订单列表" />
          {canViewRefund && <Tabs.TabPane title="退款审核" />}
        </Tabs>

        {tab === 0 && (
          <>
            {/* 搜索栏 */}
            <View className="search-bar">
              <Input
                className="search-input"
                placeholder="搜索订单号/用户名"
                value={keyword}
                onInput={(e) => setKeyword(e.detail.value)}
                onConfirm={handleSearch}
              />
              <View className="filter-row">
                {[
                  "",
                  "pending",
                  "paid",
                  "completed",
                  "cancelled",
                  "refunding",
                  "refunded",
                ].map((s) => (
                  <Tag
                    key={s}
                    color={statusFilter === s ? "primary" : "default"}
                    size="small"
                    onClick={() => {
                      setStatusFilter(s);
                      fetchOrders(1, keyword, s);
                    }}
                  >
                    {s ? STATUS_LABELS[s] || s : "全部"}
                  </Tag>
                ))}
              </View>
            </View>

            <ScrollView
              scrollY
              className="list-scroll"
              onScrollToLower={handleLoadMoreOrders}
            >
              {loading && orders.length === 0 ? (
                <View className="loading-wrapper">
                  <Loading>加载中...</Loading>
                </View>
              ) : orders.length === 0 ? (
                <Empty>
                  <Empty.Description>暂无订单数据</Empty.Description>
                </Empty>
              ) : (
                orders.map((o) => (
                  <View key={o.orderNo} className="list-card">
                    <View className="card-header">
                      <View className="card-title-row">
                        <Text className="card-name">{o.packageName}</Text>
                        <Tag
                          size="small"
                          color={
                            o.status === "completed"
                              ? "success"
                              : o.status === "cancelled"
                                ? "danger"
                                : "primary"
                          }
                        >
                          {STATUS_LABELS[o.status] || o.status}
                        </Tag>
                      </View>
                      <Text className="card-id">#{o.orderNo}</Text>
                    </View>
                    <View className="card-meta">
                      <Text className="meta-item">
                        用户: {o.username} (ID:{o.userId})
                      </Text>
                      <Text className="meta-item">实付: ¥{o.paidAmount}</Text>
                      {o.packageLevel && (
                        <Text className="meta-item">
                          等级: {o.packageLevel}
                        </Text>
                      )}
                    </View>
                    <View className="card-meta">
                      <Text className="meta-item">
                        创建: {new Date(o.createTime).toLocaleString("zh-CN")}
                      </Text>
                      {o.paidTime && (
                        <Text className="meta-item">
                          支付: {new Date(o.paidTime).toLocaleString("zh-CN")}
                        </Text>
                      )}
                    </View>
                  </View>
                ))
              )}
              {orders.length > 0 && orders.length < totalOrders && (
                <View className="load-more" onClick={handleLoadMoreOrders}>
                  <Text>加载更多</Text>
                </View>
              )}
            </ScrollView>
          </>
        )}

        {tab === 1 && canViewRefund && (
          <ScrollView
            scrollY
            className="list-scroll"
            onScrollToLower={handleLoadMoreRefunds}
          >
            {refunds.length === 0 ? (
              <Empty>
                <Empty.Description>暂无退款申请</Empty.Description>
              </Empty>
            ) : (
              refunds.map((r) => (
                <View key={r.id} className="list-card">
                  <View className="card-header">
                    <View className="card-title-row">
                      <Text className="card-name">退款单 {r.refundNo}</Text>
                      <Tag
                        size="small"
                        color={
                          r.status === "refunding"
                            ? "warning"
                            : r.status === "refunded"
                              ? "success"
                              : "danger"
                        }
                      >
                        {REFUND_STATUS_LABELS[r.status] || r.status}
                      </Tag>
                    </View>
                    <Text className="card-id">#{r.orderNo}</Text>
                  </View>
                  <View className="card-meta">
                    <Text className="meta-item">
                      用户: {r.username} (ID:{r.userId})
                    </Text>
                    <Text className="meta-item">退款: ¥{r.refundAmount}</Text>
                    <Text className="meta-item">原因: {r.reason}</Text>
                  </View>
                  <View className="card-meta">
                    <Text className="meta-item">
                      申请: {new Date(r.applyTime).toLocaleString("zh-CN")}
                    </Text>
                    {r.auditTime && (
                      <Text className="meta-item">
                        审核: {new Date(r.auditTime).toLocaleString("zh-CN")}
                      </Text>
                    )}
                  </View>
                  {r.status === "refunding" && (
                    <View className="card-actions">
                      <View
                        className="action-btn primary"
                        onClick={() => openAudit(r)}
                      >
                        通过
                      </View>
                      <View
                        className="action-btn danger"
                        onClick={() => openAudit(r)}
                      >
                        驳回
                      </View>
                    </View>
                  )}
                </View>
              ))
            )}
            {refunds.length > 0 && refunds.length < totalRefunds && (
              <View className="load-more" onClick={handleLoadMoreRefunds}>
                <Text>加载更多</Text>
              </View>
            )}
          </ScrollView>
        )}

        {/* 审核弹窗 */}
        <Popup
          open={auditPopupVisible}
          placement="bottom"
          rounded
          onClose={() => setAuditPopupVisible(false)}
        >
          <View className="popup-content">
            <View className="popup-header">
              <Text className="popup-title">退款审核</Text>
              <Text
                className="popup-close"
                onClick={() => setAuditPopupVisible(false)}
              >
                ×
              </Text>
            </View>
            <View className="popup-body">
              {auditingRefund && (
                <>
                  <View className="audit-info">
                    <Text className="audit-line">
                      订单号: {auditingRefund.orderNo}
                    </Text>
                    <Text className="audit-line">
                      退款金额: ¥{auditingRefund.refundAmount}
                    </Text>
                    <Text className="audit-line">
                      退款原因: {auditingRefund.reason}
                    </Text>
                  </View>
                  <View className="form-item">
                    <Text className="form-label">审核备注</Text>
                    <Input
                      className="form-input"
                      placeholder="请输入审核备注"
                      value={auditRemark}
                      onInput={(e) => setAuditRemark(e.detail.value)}
                    />
                  </View>
                  <View className="audit-actions">
                    <View
                      className="action-btn danger"
                      onClick={handleRejectRefund}
                    >
                      驳回退款
                    </View>
                    <View
                      className="action-btn primary"
                      onClick={handleApproveRefund}
                    >
                      通过退款
                    </View>
                  </View>
                </>
              )}
            </View>
          </View>
        </Popup>
      </View>
    </PageLayout>
  );
};

export default OrderManagePage;
