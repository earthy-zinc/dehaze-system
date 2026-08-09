import React, { useState, useCallback, useEffect, useRef } from "react";
import { View, Text, ScrollView } from "@tarojs/components";
import Taro, {
  useLoad,
  usePullDownRefresh,
  useReachBottom,
} from "@tarojs/taro";
import { Loading, Empty, Button, Tag } from "@taroify/core";
import { TaskAPI } from "dehaze-sdk-js";
import type { TaskVO, TaskQuery, TaskStatus } from "dehaze-sdk-js";
import PageLayout from "@/layout";
import ErrorState from "@/components/common/ErrorState";
import { getErrorMessage } from "@/utils/error";
import { confirmDialog } from "@/utils/dialog";
import { usePermission } from "@/hooks/usePermission";
import {
  STATUS_FILTERS,
  PAGE_SIZE,
  STATUS_TAG,
  TASK_TYPE_LABEL,
  shortTaskId,
} from "@/pages/task/constants";
import "./index.less";

const TaskManagePage: React.FC = () => {
  const { hasPermission } = usePermission();
  const canCancel = hasPermission("sys:task:cancel");
  const canRetry = hasPermission("sys:task:retry");

  const [taskList, setTaskList] = useState<TaskVO[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<"" | TaskStatus>("");
  const [pageNum, setPageNum] = useState(1);
  const [hasMore, setHasMore] = useState(true);

  const [cancelLoadingId, setCancelLoadingId] = useState<string | null>(null);
  const [retryLoadingId, setRetryLoadingId] = useState<string | null>(null);

  const pollingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const taskListRef = useRef<TaskVO[]>([]);
  useEffect(() => {
    taskListRef.current = taskList;
  }, [taskList]);

  const loadTaskList = useCallback(async (status: "" | TaskStatus) => {
    setLoading(true);
    setLoadError(null);
    try {
      const query: TaskQuery = {
        pageNum: 1,
        pageSize: PAGE_SIZE,
        status: status || undefined,
      };
      const res = await TaskAPI.getPage(query);
      const list = res.list || [];
      setTaskList(list);
      setPageNum(1);
      setHasMore(list.length < (res.total || 0));
    } catch (err: unknown) {
      setLoadError(getErrorMessage(err, "加载失败，请重试"));
    } finally {
      setLoading(false);
    }
  }, []);

  const loadMore = useCallback(async () => {
    if (loading || !hasMore) return;
    const nextPage = pageNum + 1;
    try {
      const query: TaskQuery = {
        pageNum: nextPage,
        pageSize: PAGE_SIZE,
        status: statusFilter || undefined,
      };
      const res = await TaskAPI.getPage(query);
      const list = res.list || [];
      setTaskList((prev) => [...prev, ...list]);
      setPageNum(nextPage);
      setHasMore(list.length >= PAGE_SIZE);
    } catch {
      Taro.showToast({ title: "加载更多失败", icon: "none" });
    }
  }, [loading, hasMore, pageNum, statusFilter]);

  const stopPolling = useCallback(() => {
    if (pollingTimerRef.current) {
      clearInterval(pollingTimerRef.current);
      pollingTimerRef.current = null;
    }
  }, []);

  const pollOnce = useCallback(async () => {
    const activeTasks = taskListRef.current.filter(
      (t) => t.status === 1 || t.status === 2
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
    } catch {
      // 轮询失败静默处理
    }
  }, [stopPolling]);

  const startPolling = useCallback(() => {
    if (pollingTimerRef.current) return;
    const hasActive = taskListRef.current.some(
      (t) => t.status === 1 || t.status === 2
    );
    if (!hasActive) return;
    pollingTimerRef.current = setInterval(pollOnce, 3000);
  }, [pollOnce]);

  useEffect(() => {
    const hasActive = taskList.some((t) => t.status === 1 || t.status === 2);
    if (hasActive) startPolling();
    else stopPolling();
  }, [taskList, startPolling, stopPolling]);

  useEffect(() => () => stopPolling(), [stopPolling]);

  useLoad(() => {
    loadTaskList(statusFilter);
  });

  usePullDownRefresh(() => {
    loadTaskList(statusFilter).finally(() => Taro.stopPullDownRefresh());
  });

  useReachBottom(() => {
    loadMore();
  });

  const handleStatusChange = useCallback(
    (status: "" | TaskStatus) => {
      setStatusFilter(status);
      stopPolling();
      loadTaskList(status);
    },
    [stopPolling, loadTaskList]
  );

  const handleCancel = useCallback(
    async (task: TaskVO) => {
      if (!canCancel) return;
      const confirmed = await confirmDialog({
        title: "取消任务",
        content: "确认取消该任务吗？",
        confirmColor: "#ff4d4f",
      });
      if (!confirmed) return;
      setCancelLoadingId(task.taskId);
      try {
        await TaskAPI.cancel(task.taskId);
        setTaskList((prev) =>
          prev.map((t) =>
            t.taskId === task.taskId ? { ...t, status: 5 as TaskStatus } : t
          )
        );
        Taro.showToast({ title: "任务已取消", icon: "success" });
      } catch (err: unknown) {
        Taro.showToast({
          title: getErrorMessage(err, "取消失败"),
          icon: "none",
        });
      } finally {
        setCancelLoadingId(null);
      }
    },
    [canCancel]
  );

  const handleRetry = useCallback(
    async (task: TaskVO) => {
      if (!canRetry) return;
      setRetryLoadingId(task.taskId);
      try {
        await TaskAPI.retry(task.taskId);
        setTaskList((prev) =>
          prev.map((t) =>
            t.taskId === task.taskId ? { ...t, status: 1 as TaskStatus } : t
          )
        );
        Taro.showToast({ title: "已重新提交", icon: "success" });
      } catch (err: unknown) {
        Taro.showToast({
          title: getErrorMessage(err, "重试失败"),
          icon: "none",
        });
      } finally {
        setRetryLoadingId(null);
      }
    },
    [canRetry]
  );

  return (
    <PageLayout level="L2" title="任务管理">
      <View className="task-manage-page">
        <ScrollView
          scrollX
          className="filter-bar"
          enhanced
          showScrollbar={false}
        >
          {STATUS_FILTERS.map((filter) => (
            <View
              key={String(filter.value)}
              className={`filter-item ${statusFilter === filter.value ? "active" : ""}`}
              onClick={() => handleStatusChange(filter.value)}
            >
              <Text>{filter.label}</Text>
            </View>
          ))}
        </ScrollView>

        <ScrollView scrollY className="task-list">
          {loading && taskList.length === 0 ? (
            <View className="loading-wrapper">
              <Loading>加载中...</Loading>
            </View>
          ) : loadError && taskList.length === 0 ? (
            <ErrorState
              message={loadError}
              onRetry={() => loadTaskList(statusFilter)}
            />
          ) : taskList.length === 0 ? (
            <Empty>
              <Empty.Description>暂无任务</Empty.Description>
            </Empty>
          ) : (
            <>
              {taskList.map((task) => {
                const tagInfo = STATUS_TAG[task.status] || {
                  label: String(task.status),
                  color: "default" as const,
                };
                const typeLabel = task.taskType
                  ? TASK_TYPE_LABEL[task.taskType] || task.taskType
                  : "未知类型";
                const isFailed = task.status === 4;
                const isActive = task.status === 1 || task.status === 2;
                return (
                  <View key={task.taskId} className="task-card">
                    <View className="task-header">
                      <Tag color={tagInfo.color} size="small">
                        {tagInfo.label}
                      </Tag>
                      <Text className="task-id">
                        {shortTaskId(task.taskId)}
                      </Text>
                    </View>
                    <View className="task-body">
                      <Text className="task-type">{typeLabel}</Text>
                      {task.progress !== undefined && isActive && (
                        <View className="task-progress">
                          <View className="progress-bar">
                            <View
                              className="progress-fill"
                              style={{ width: `${task.progress}%` }}
                            />
                          </View>
                          <Text className="progress-text">
                            {task.progress}%
                          </Text>
                        </View>
                      )}
                      {task.error && (
                        <Text className="task-error">{task.error}</Text>
                      )}
                    </View>
                    <View className="task-footer">
                      <Text className="task-time">
                        {task.createdAt
                          ? new Date(task.createdAt).toLocaleString("zh-CN")
                          : "-"}
                      </Text>
                      <View className="task-actions">
                        {isActive && canCancel && (
                          <Button
                            size="mini"
                            color="danger"
                            loading={cancelLoadingId === task.taskId}
                            onClick={() => handleCancel(task)}
                          >
                            取消
                          </Button>
                        )}
                        {isFailed && canRetry && (
                          <Button
                            size="mini"
                            color="primary"
                            loading={retryLoadingId === task.taskId}
                            onClick={() => handleRetry(task)}
                          >
                            重试
                          </Button>
                        )}
                      </View>
                    </View>
                  </View>
                );
              })}
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
      </View>
    </PageLayout>
  );
};

export default TaskManagePage;
