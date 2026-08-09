import React, { useState, useEffect, useCallback } from "react";
import { View, Text, ScrollView } from "@tarojs/components";
import Taro from "@tarojs/taro";
import { Tag } from "@taroify/core";
import { OrderAPI } from "dehaze-sdk-js";
import type { MyOrderVO, OrderStatus, MyOrderQuery } from "dehaze-sdk-js";
import PageLayout from "@/layout";
import { confirmDialog } from "@/utils/dialog";
import "./index.less";

const STATUS_LABELS: Record<string, string> = {
  pending: "待支付",
  paid: "已支付",
  completed: "已完成",
  cancelled: "已取消",
  refunding: "退款中",
  refunded: "已退款",
};

const STATUS_FILTERS: { label: string; value: string }[] = [
  { label: "全部", value: "" },
  { label: "待支付", value: "pending" },
  { label: "已支付", value: "paid" },
  { label: "已完成", value: "completed" },
  { label: "已取消", value: "cancelled" },
];

const PAGE_SIZE = 20;

const OrdersPage: React.FC = () => {
  const [orders, setOrders] = useState<MyOrderVO[]>([]);
  const [loading, setLoading] = useState(false);
  const [total, setTotal] = useState(0);
  const [pageNum, setPageNum] = useState(1);
  const [statusFilter, setStatusFilter] = useState("");

  const loadOrders = useCallback(async (page: number, status: string) => {
    setLoading(true);
    try {
      const params: MyOrderQuery = { pageNum: page, pageSize: PAGE_SIZE };
      if (status) params.status = status as OrderStatus;
      const res = await OrderAPI.listMy(params);
      setOrders(res.list || []);
      setTotal(res.total);
      setPageNum(page);
    } catch {
      Taro.showToast({ title: "加载失败", icon: "none" });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadOrders(1, "");
  }, [loadOrders]);

  const handleFilter = (status: string) => {
    setStatusFilter(status);
    loadOrders(1, status);
  };

  const handleLoadMore = () => {
    if (orders.length < total) {
      loadOrders(pageNum + 1, statusFilter);
    }
  };

  const handleCancel = async (order: MyOrderVO) => {
    const confirmed = await confirmDialog({
      title: "取消订单",
      content: `确定要取消订单「${order.orderNo}」吗？`,
      confirmColor: "#ef4444",
    });
    if (!confirmed) return;
    try {
      await OrderAPI.cancel(order.orderNo, "用户主动取消");
      Taro.showToast({ title: "订单已取消", icon: "success" });
      loadOrders(1, statusFilter);
    } catch {
      Taro.showToast({ title: "取消失败", icon: "none" });
    }
  };

  const handlePay = async (order: MyOrderVO) => {
    try {
      const res = await OrderAPI.pay(order.orderNo, { payMethod: "wechat" });
      if (res.paid) {
        Taro.showToast({ title: "支付成功", icon: "success" });
        loadOrders(1, statusFilter);
      } else if (res.payUrl) {
        Taro.showToast({ title: "支付功能开发中", icon: "none" });
      }
    } catch {
      Taro.showToast({ title: "支付失败", icon: "none" });
    }
  };

  const formatTime = (timeStr: string) => {
    return new Date(timeStr).toLocaleString("zh-CN");
  };

  return (
    <PageLayout level="L2" title="我的订单">
      <View className="personal-orders-page">
        {/* 状态筛选 */}
        <View className="filter-row">
          {STATUS_FILTERS.map((f) => (
            <Tag
              key={f.value}
              color={statusFilter === f.value ? "primary" : "default"}
              size="small"
              onClick={() => handleFilter(f.value)}
            >
              {f.label}
            </Tag>
          ))}
        </View>

        <ScrollView scrollY className="orders-scroll" onScrollToLower={handleLoadMore}>
          {loading && orders.length === 0 ? (
            <View className="loading-wrapper">
              <Text>加载中...</Text>
            </View>
          ) : orders.length === 0 ? (
            <View className="empty-wrapper">
              <Text className="empty-icon">🛒</Text>
              <Text className="empty-title">暂无订单</Text>
              <Text className="empty-desc">您购买的套餐订单将显示在这里</Text>
            </View>
          ) : (
            <>
              {orders.map((order) => (
                <View key={order.orderNo} className="order-card">
                  <View className="order-header">
                    <Text className="order-no">#{order.orderNo}</Text>
                    <Text className={`order-status status-${order.status}`}>
                      {STATUS_LABELS[order.status] || order.status}
                    </Text>
                  </View>
                  <View className="order-body">
                    <Text className="order-package">{order.packageName}</Text>
                    <Text className="order-amount">¥{order.payableAmount}</Text>
                  </View>
                  <View className="order-footer">
                    <Text className="order-time">{formatTime(order.createTime)}</Text>
                    {order.paidTime && (
                      <Text className="order-time">支付: {formatTime(order.paidTime)}</Text>
                    )}
                  </View>
                  {(order.status === "pending" || order.status === "paid") && (
                    <View className="order-actions">
                      {order.status === "pending" && (
                        <>
                          <View className="action-btn primary" onClick={() => handlePay(order)}>
                            立即支付
                          </View>
                          <View className="action-btn danger" onClick={() => handleCancel(order)}>
                            取消订单
                          </View>
                        </>
                      )}
                      {order.status === "paid" && (
                        <View className="action-btn primary" onClick={() => handleCancel(order)}>
                          申请退款
                        </View>
                      )}
                    </View>
                  )}
                </View>
              ))}
              {orders.length > 0 && orders.length < total && (
                <View className="load-more" onClick={handleLoadMore}>
                  <Text>加载更多</Text>
                </View>
              )}
            </>
          )}
        </ScrollView>
      </View>
    </PageLayout>
  );
};

export default OrdersPage;
