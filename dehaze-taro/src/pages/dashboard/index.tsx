import React, { useState, useEffect, useCallback } from "react";
import { View, Text, ScrollView } from "@tarojs/components";
import Taro from "@tarojs/taro";
import { Loading, Tag } from "@taroify/core";
import { AlgorithmAPI, DatasetAPI, TaskAPI, UserAPI } from "dehaze-sdk-js";
import type { TaskVO } from "dehaze-sdk-js";
import { useGlobalContext } from "@/stores/global";
import PageLayout from "@/layout";
import "./index.less";

// ==================== 常量 ====================

/** 任务状态标签映射 */
const TASK_STATUS_TAG: Record<
  string,
  { label: string; color: "default" | "primary" | "success" | "danger" }
> = {
  PENDING: { label: "待执行", color: "primary" },
  PROCESSING: { label: "执行中", color: "primary" },
  COMPLETED: { label: "已完成", color: "success" },
  FAILED: { label: "失败", color: "danger" },
  CANCELLED: { label: "已取消", color: "default" },
};

/** 核心业务流程入口 */
const WORKFLOW_ENTRIES = [
  {
    icon: "📷",
    title: "图像输入",
    desc: "上传或拍摄图片",
    route: "/pages/image-input/index",
  },
  {
    icon: "💡",
    title: "算法选择",
    desc: "选择去雾算法",
    route: "/pages/algorithm-select/index",
  },
  {
    icon: "⚙️",
    title: "去雾处理",
    desc: "执行去雾处理",
    route: "/pages/processing/index",
  },
  {
    icon: "📊",
    title: "效果对比",
    desc: "对比处理效果",
    route: "/pages/side-by-side/index",
  },
];

/** 管理入口 */
const MANAGEMENT_ENTRIES = [
  { icon: "👥", title: "用户管理", route: "/pages/system/user/index" },
  { icon: "🛡️", title: "角色管理", route: "/pages/system/role/index" },
  { icon: "📁", title: "数据集管理", route: "/pages/dataset/index" },
  { icon: "🔧", title: "算法管理", route: "/pages/algorithm/index" },
  { icon: "📋", title: "任务中心", route: "/pages/task/index" },
];

// ==================== 统计数据类型 ====================

interface Stats {
  userCount: number | null;
  datasetCount: number | null;
  algorithmCount: number | null;
  taskTotal: number | null;
}

// ==================== 页面组件 ====================

const Dashboard: React.FC = () => {
  const { state } = useGlobalContext();
  const user = state.auth.user;
  const roles = state.auth.roles || [];

  const [stats, setStats] = useState<Stats>({
    userCount: null,
    datasetCount: null,
    algorithmCount: null,
    taskTotal: null,
  });
  const [recentTasks, setRecentTasks] = useState<TaskVO[]>([]);
  const [loading, setLoading] = useState(true);

  // ==================== 数据加载 ====================

  const fetchDashboardData = useCallback(async () => {
    setLoading(true);

    // 并行请求统计数据
    const results = await Promise.allSettled([
      UserAPI.getPage({ pageNum: 1, pageSize: 1 }),
      DatasetAPI.getList({ pageNum: 1, pageSize: 1 }),
      AlgorithmAPI.getList(),
      TaskAPI.getPage({ pageNum: 1, pageSize: 5 }),
    ]);

    const newStats: Stats = {
      userCount: null,
      datasetCount: null,
      algorithmCount: null,
      taskTotal: null,
    };

    if (results[0].status === "fulfilled") {
      newStats.userCount = results[0].value.total ?? null;
    }
    if (results[1].status === "fulfilled") {
      newStats.datasetCount = results[1].value.total ?? null;
    }
    if (results[2].status === "fulfilled") {
      const tree = results[2].value || [];
      // 递归统计算法节点总数
      const countNodes = (nodes: typeof tree): number =>
        nodes.reduce(
          (sum, n) => sum + 1 + (n.children ? countNodes(n.children) : 0),
          0
        );
      newStats.algorithmCount = countNodes(tree);
    }
    if (results[3].status === "fulfilled") {
      newStats.taskTotal = results[3].value.total ?? null;
      setRecentTasks((results[3].value.list as unknown as TaskVO[]) || []);
    }

    setStats(newStats);
    setLoading(false);
  }, []);

  useEffect(() => {
    fetchDashboardData();
  }, [fetchDashboardData]);

  // ==================== 事件处理 ====================

  const handleNavigate = useCallback((route: string) => {
    Taro.navigateTo({
      url: route,
      fail: () => {
        Taro.showToast({ title: "页面开发中", icon: "none" });
      },
    });
  }, []);

  const handleRefresh = useCallback(() => {
    fetchDashboardData();
  }, [fetchDashboardData]);

  // ==================== 渲染 ====================

  /** 渲染统计卡片 */
  const renderStatCard = (
    label: string,
    value: number | null,
    color: string
  ) => (
    <View className="stat-card" key={label}>
      <Text className="stat-value" style={{ color }}>
        {value === null ? "-" : String(value)}
      </Text>
      <Text className="stat-label">{label}</Text>
    </View>
  );

  return (
    <PageLayout showTabbar={false} title="工作台">
      <View className="dashboard-page">
        <ScrollView scrollY className="dashboard-scroll">
          {/* 用户欢迎区 */}
          <View className="welcome-section">
            <View className="user-avatar">
              {user?.avatar ? (
                <Text className="avatar-text">
                  {user.nickname?.[0] || user.username?.[0] || "U"}
                </Text>
              ) : (
                <Text className="avatar-text">
                  {user?.nickname?.[0] || user?.username?.[0] || "U"}
                </Text>
              )}
            </View>
            <View className="user-info">
              <Text className="user-name">
                {user?.nickname || user?.username || "用户"}
              </Text>
              <View className="user-roles">
                {roles.length > 0 ? (
                  roles.slice(0, 3).map((role) => (
                    <Tag key={role} size="small" color="primary">
                      {role.replace("ROLE_", "")}
                    </Tag>
                  ))
                ) : (
                  <Text className="user-role-text">暂无角色</Text>
                )}
              </View>
            </View>
            <Text className="refresh-btn" onClick={handleRefresh}>
              刷新
            </Text>
          </View>

          {/* 统计概览 */}
          <View className="stats-grid">
            {renderStatCard("用户总数", stats.userCount, "#1890ff")}
            {renderStatCard("数据集", stats.datasetCount, "#52c41a")}
            {renderStatCard("算法数", stats.algorithmCount, "#722ed1")}
            {renderStatCard("任务总数", stats.taskTotal, "#fa8c16")}
          </View>

          {/* 核心业务流程 */}
          <View className="section">
            <Text className="section-title">核心流程</Text>
            <View className="entry-grid">
              {WORKFLOW_ENTRIES.map((entry) => (
                <View
                  key={entry.route}
                  className="entry-card workflow-card"
                  onClick={() => handleNavigate(entry.route)}
                >
                  <Text className="entry-icon">{entry.icon}</Text>
                  <Text className="entry-title">{entry.title}</Text>
                  <Text className="entry-desc">{entry.desc}</Text>
                </View>
              ))}
            </View>
          </View>

          {/* 管理入口 */}
          <View className="section">
            <Text className="section-title">系统管理</Text>
            <View className="management-list">
              {MANAGEMENT_ENTRIES.map((entry) => (
                <View
                  key={entry.route}
                  className="management-item"
                  onClick={() => handleNavigate(entry.route)}
                >
                  <Text className="management-icon">{entry.icon}</Text>
                  <Text className="management-title">{entry.title}</Text>
                  <Text className="management-arrow">›</Text>
                </View>
              ))}
            </View>
          </View>

          {/* 最近任务 */}
          <View className="section">
            <View className="section-header">
              <Text className="section-title">最近任务</Text>
              <Text
                className="section-more"
                onClick={() => handleNavigate("/pages/task/index")}
              >
                查看全部 ›
              </Text>
            </View>
            {loading ? (
              <View className="loading-wrapper">
                <Loading>加载中...</Loading>
              </View>
            ) : recentTasks.length === 0 ? (
              <View className="empty-tasks">
                <Text>暂无任务记录</Text>
              </View>
            ) : (
              <View className="task-list">
                {recentTasks.map((task) => {
                  const tagInfo = TASK_STATUS_TAG[task.status] || {
                    label: task.status,
                    color: "default" as const,
                  };
                  return (
                    <View
                      key={task.taskId}
                      className="recent-task"
                      onClick={() => handleNavigate("/pages/task/index")}
                    >
                      <View className="task-info">
                        <Tag color={tagInfo.color} size="small">
                          {tagInfo.label}
                        </Tag>
                        <Text className="task-type">
                          {task.taskType || "未知类型"}
                        </Text>
                      </View>
                      <Text className="task-time">
                        {task.createdAt
                          ? new Date(task.createdAt).toLocaleDateString("zh-CN")
                          : "-"}
                      </Text>
                    </View>
                  );
                })}
              </View>
            )}
          </View>

          <View className="dashboard-footer">
            <Text>图像去雾系统 v1.0</Text>
          </View>
        </ScrollView>
      </View>
    </PageLayout>
  );
};

export default Dashboard;
