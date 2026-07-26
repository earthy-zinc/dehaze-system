import React, { useState, useEffect, useCallback, useRef } from "react";
import { View, Text, ScrollView } from "@tarojs/components";
import Taro, {
  useLoad,
  usePullDownRefresh,
  useReachBottom,
} from "@tarojs/taro";
import { Navbar, Loading, Empty } from "@taroify/core";
import { ArrowLeft } from "@taroify/icons";
import { TaskAPI } from "dehaze-sdk-js";
import { confirmDialog } from "@/utils/dialog";
import type { TaskCategory, TaskVO, TaskQuery, TaskStatus } from "dehaze-sdk-js";
import ErrorState from "@/components/common/ErrorState";
import { getErrorMessage } from "@/utils/error";
import {
  POLLING_INTERVAL,
  POLLING_STATUSES,
  STATUS_FILTERS,
  CATEGORY_FILTERS,
  PAGE_SIZE,
} from "./constants";
import TaskCard from "./components/TaskCard";
import TaskDetailPopup from "./components/TaskDetailPopup";
import "./index.less";

// ==================== 页面组件 ====================

const TaskPage: React.FC = () => {
  const [taskList, setTaskList] = useState<TaskVO[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<"" | TaskStatus>("");
  const [categoryFilter, setCategoryFilter] = useState<"" | TaskCategory>("");
  const [pageNum, setPageNum] = useState(1);
  const [hasMore, setHasMore] = useState(true);

  // 任务详情弹窗
  const [detailTask, setDetailTask] = useState<TaskVO | null>(null);
  const [detailVisible, setDetailVisible] = useState(false);

  // 操作加载状态
  const [cancelLoadingId, setCancelLoadingId] = useState<string | null>(null);
  const [downloadLoadingId, setDownloadLoadingId] = useState<string | null>(
    null
  );

  // 轮询定时器
  const pollingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // 最新任务列表引用（供轮询回调读取）
  const taskListRef = useRef<TaskVO[]>([]);
  useEffect(() => {
    taskListRef.current = taskList;
  }, [taskList]);

  // ==================== 数据加载 ====================

  /** 加载任务列表（第一页） */
  const loadTaskList = useCallback(
    async (status: "" | TaskStatus, category: "" | TaskCategory) => {
      setLoading(true);
      setLoadError(null);
      try {
        const query: TaskQuery = {
          pageNum: 1,
          pageSize: PAGE_SIZE,
          status: status || undefined,
          taskCategory: category || undefined,
        };
        const res = await TaskAPI.getPage(query);
        const list = (res.list as unknown as TaskVO[]) || [];
        setTaskList(list);
        setPageNum(1);
        setHasMore(list.length < (res.total || 0));
      } catch (err: unknown) {
        setLoadError(getErrorMessage(err, "加载失败，请重试"));
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  /** 加载更多（下一页） */
  const loadMore = useCallback(async () => {
    if (loading || !hasMore) return;
    const nextPage = pageNum + 1;
    try {
      const query: TaskQuery = {
        pageNum: nextPage,
        pageSize: PAGE_SIZE,
        status: statusFilter || undefined,
        taskCategory: categoryFilter || undefined,
      };
      const res = await TaskAPI.getPage(query);
      const list = (res.list as unknown as TaskVO[]) || [];
      setTaskList((prev) => [...prev, ...list]);
      setPageNum(nextPage);
      setHasMore(list.length >= PAGE_SIZE);
    } catch (err: unknown) {
      Taro.showToast({
        title: getErrorMessage(err, "加载更多失败"),
        icon: "none",
      });
    }
  }, [loading, hasMore, pageNum, statusFilter, categoryFilter]);

  // ==================== 轮询逻辑 ====================

  /** 停止轮询 */
  const stopPolling = useCallback(() => {
    if (pollingTimerRef.current) {
      clearInterval(pollingTimerRef.current);
      pollingTimerRef.current = null;
    }
  }, []);

  /** 执行一次轮询：查询所有 PENDING/PROCESSING 任务状态 */
  const pollOnce = useCallback(async () => {
    const activeTasks = taskListRef.current.filter((t) =>
      POLLING_STATUSES.includes(t.status)
    );
    if (activeTasks.length === 0) {
      stopPolling();
      return;
    }
    try {
      const results = await Promise.all(
        activeTasks.map((t) => TaskAPI.getStatus(t.taskId))
      );
      setTaskList((prev) =>
        prev.map((task) => {
          const updated = results.find((r) => r.taskId === task.taskId);
          return updated || task;
        })
      );
      // 同步更新详情弹窗中的任务
      setDetailTask((prev) => {
        if (!prev) return prev;
        const updated = results.find((r) => r.taskId === prev.taskId);
        return updated || prev;
      });
    } catch {
      // 轮询失败静默处理
    }
  }, [stopPolling]);

  /** 启动轮询 */
  const startPolling = useCallback(() => {
    if (pollingTimerRef.current) return;
    const hasActive = taskListRef.current.some((t) =>
      POLLING_STATUSES.includes(t.status)
    );
    if (!hasActive) return;
    pollingTimerRef.current = setInterval(pollOnce, POLLING_INTERVAL);
  }, [pollOnce]);

  // 任务列表变化后，检查是否需要启动/停止轮询
  useEffect(() => {
    const hasActive = taskList.some((t) => POLLING_STATUSES.includes(t.status));
    if (hasActive) {
      startPolling();
    } else {
      stopPolling();
    }
  }, [taskList, startPolling, stopPolling]);

  // 组件卸载时清除定时器
  useEffect(() => {
    return stopPolling;
  }, [stopPolling]);

  // ==================== 生命周期 ====================

  useLoad(() => {
    loadTaskList(statusFilter, categoryFilter);
  });

  usePullDownRefresh(() => {
    loadTaskList(statusFilter, categoryFilter).finally(() => {
      Taro.stopPullDownRefresh();
    });
  });

  useReachBottom(() => {
    loadMore();
  });

  // ==================== 事件处理 ====================

  /** 状态筛选变化 */
  const handleStatusChange = useCallback(
    (status: "" | TaskStatus) => {
      setStatusFilter(status);
      stopPolling();
      loadTaskList(status, categoryFilter);
    },
    [stopPolling, loadTaskList, categoryFilter],
  );

  /** 任务类别筛选变化 */
  const handleCategoryChange = useCallback(
    (category: "" | TaskCategory) => {
      setCategoryFilter(category);
      stopPolling();
      loadTaskList(statusFilter, category);
    },
    [stopPolling, loadTaskList, statusFilter],
  );

  /** 查看任务详情 */
  const handleDetail = useCallback((task: TaskVO) => {
    setDetailTask(task);
    setDetailVisible(true);
  }, []);

  /** 关闭详情弹窗 */
  const handleDetailClose = useCallback(() => {
    setDetailVisible(false);
  }, []);

  /** 取消任务（二次确认） */
  const handleCancel = useCallback(async (task: TaskVO) => {
    const confirmed = await confirmDialog({
      title: "取消任务",
      content: "确认取消该任务吗？取消后不可恢复。",
      confirmColor: "#ff4d4f",
    });
    if (!confirmed) return;
    setCancelLoadingId(task.taskId);
    try {
      await TaskAPI.cancel(task.taskId);
      // 本地更新状态
      const now = new Date().toISOString();
      const updated: TaskVO = {
        ...task,
        status: "CANCELLED",
        completedAt: now,
      };
      setTaskList((prev) =>
        prev.map((t) => (t.taskId === task.taskId ? updated : t))
      );
      setDetailTask((prev) => (prev?.taskId === task.taskId ? updated : prev));
      Taro.showToast({ title: "任务已取消", icon: "success" });
    } catch (err: unknown) {
      Taro.showToast({ title: getErrorMessage(err, "取消失败"), icon: "none" });
    } finally {
      setCancelLoadingId(null);
    }
  }, []);

  /** 下载任务结果 */
  const handleDownload = useCallback(async (task: TaskVO) => {
    if (task.status !== "COMPLETED") {
      Taro.showToast({ title: "任务尚未完成", icon: "none" });
      return;
    }
    setDownloadLoadingId(task.taskId);
    try {
      // 获取最新任务状态以拿到 downloadUrl
      const latest = await TaskAPI.getStatus(task.taskId);
      if (!latest.downloadUrl) {
        Taro.showToast({ title: "下载链接不存在", icon: "none" });
        return;
      }
      if (latest.expiresAt && new Date(latest.expiresAt) < new Date()) {
        Taro.showToast({ title: "任务结果已过期", icon: "none" });
        return;
      }
      // 下载文件
      const downloadRes = await Taro.downloadFile({ url: latest.downloadUrl });
      if (process.env.TARO_ENV === "h5") {
        // H5 端：openDocument 不可用，通过新标签页打开下载
        window.open(downloadRes.tempFilePath, "_blank");
      } else {
        // 小程序端：使用 openDocument 打开文件
        await Taro.openDocument({
          filePath: downloadRes.tempFilePath,
          showMenu: true,
        });
      }
    } catch (err: unknown) {
      Taro.showToast({ title: getErrorMessage(err, "下载失败"), icon: "none" });
    } finally {
      setDownloadLoadingId(null);
    }
  }, []);

  // ==================== 渲染 ====================

  return (
    <View className="task-page">
      <Navbar title="任务中心">
        <Navbar.NavLeft>
          <ArrowLeft />
        </Navbar.NavLeft>
      </Navbar>

      {/* 任务类别筛选 */}
      <View className="category-bar">
        {CATEGORY_FILTERS.map((filter) => (
          <View
            key={filter.value}
            className={`category-item ${categoryFilter === filter.value ? "active" : ""}`}
            onClick={() => handleCategoryChange(filter.value)}
          >
            <Text>{filter.label}</Text>
          </View>
        ))}
      </View>

      {/* 状态筛选 */}
      <ScrollView scrollX className="filter-bar" enhanced showScrollbar={false}>
        {STATUS_FILTERS.map((filter) => (
          <View
            key={filter.value}
            className={`filter-item ${statusFilter === filter.value ? "active" : ""}`}
            onClick={() => handleStatusChange(filter.value)}
          >
            <Text>{filter.label}</Text>
          </View>
        ))}
      </ScrollView>

      {/* 任务列表 */}
      <ScrollView scrollY className="task-list">
        {loading && taskList.length === 0 ? (
          <View className="loading-wrapper">
            <Loading>加载中...</Loading>
          </View>
        ) : loadError && taskList.length === 0 ? (
          <ErrorState
            message={loadError}
            onRetry={() => loadTaskList(statusFilter, categoryFilter)}
          />
        ) : taskList.length === 0 ? (
          <Empty>
            <Empty.Description>暂无任务</Empty.Description>
          </Empty>
        ) : (
          <>
            {taskList.map((task) => (
              <TaskCard
                key={task.taskId}
                task={task}
                cancelLoading={cancelLoadingId === task.taskId}
                downloadLoading={downloadLoadingId === task.taskId}
                onClick={handleDetail}
                onCancel={handleCancel}
                onDownload={handleDownload}
              />
            ))}
            {hasMore ? (
              <View className="load-more" onClick={loadMore}>
                <Text>加载更多</Text>
              </View>
            ) : (
              <View className="no-more">
                <Text>没有更多了</Text>
              </View>
            )}
          </>
        )}
      </ScrollView>

      {/* 任务详情弹窗 */}
      <TaskDetailPopup
        open={detailVisible}
        task={detailTask}
        cancelLoading={
          detailTask ? cancelLoadingId === detailTask.taskId : false
        }
        downloadLoading={
          detailTask ? downloadLoadingId === detailTask.taskId : false
        }
        onClose={handleDetailClose}
        onCancel={handleCancel}
        onDownload={handleDownload}
      />
    </View>
  );
};

export default TaskPage;
