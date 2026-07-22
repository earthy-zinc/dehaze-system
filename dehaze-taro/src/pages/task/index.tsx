import React, { useState, useEffect, useCallback, useRef } from "react";
import { View, Text, ScrollView } from "@tarojs/components";
import Taro, {
  useLoad,
  usePullDownRefresh,
  useReachBottom,
} from "@tarojs/taro";
import {
  Navbar,
  Loading,
  Empty,
  Tag,
  Button,
  Popup,
  Progress,
} from "@taroify/core";
import { ArrowLeft } from "@taroify/icons";
import { TaskAPI } from "dehaze-sdk-js";
import type { TaskVO, TaskQuery, TaskStatus } from "dehaze-sdk-js";
import ErrorState from "@/components/common/ErrorState";
import "./index.less";

// ==================== 常量定义 ====================

/** 轮询间隔（毫秒） */
const POLLING_INTERVAL = 3000;

/** 需要轮询的任务状态 */
const POLLING_STATUSES: TaskStatus[] = ["PENDING", "PROCESSING"];

/** 终态状态集合 */
const TERMINAL_STATUSES: TaskStatus[] = ["COMPLETED", "FAILED", "CANCELLED"];

/** 状态筛选选项 */
const STATUS_FILTERS: { label: string; value: "" | TaskStatus }[] = [
  { label: "全部", value: "" },
  { label: "待执行", value: "PENDING" },
  { label: "执行中", value: "PROCESSING" },
  { label: "已完成", value: "COMPLETED" },
  { label: "失败", value: "FAILED" },
  { label: "已取消", value: "CANCELLED" },
];

/** 状态标签映射 */
const STATUS_TAG: Record<
  TaskStatus,
  { label: string; color: "default" | "primary" | "success" | "danger" }
> = {
  PENDING: { label: "待执行", color: "primary" },
  PROCESSING: { label: "执行中", color: "primary" },
  COMPLETED: { label: "已完成", color: "success" },
  FAILED: { label: "失败", color: "danger" },
  CANCELLED: { label: "已取消", color: "default" },
};

/** 任务类型映射 */
const TASK_TYPE_LABEL: Record<string, string> = {
  dataset_export: "数据集导出",
  item_download: "数据项下载",
  batch_download: "批量下载",
  custom_export: "自定义导出",
};

/** 每页条数 */
const PAGE_SIZE = 10;

// ==================== 工具函数 ====================

/** 格式化时间 */
function formatTime(t?: string): string {
  if (!t) return "-";
  const d = new Date(t);
  if (Number.isNaN(d.getTime())) return String(t);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

/** 截断任务ID */
function shortTaskId(taskId: string): string {
  if (taskId.length <= 16) return taskId;
  return `${taskId.slice(0, 8)}...${taskId.slice(-6)}`;
}

// ==================== 页面组件 ====================

const TaskPage: React.FC = () => {
  const [taskList, setTaskList] = useState<TaskVO[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<"" | TaskStatus>("");
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
      const list = (res.list as unknown as TaskVO[]) || [];
      setTaskList(list);
      setPageNum(1);
      setHasMore(list.length < (res.total || 0));
    } catch (err: any) {
      setLoadError(err?.message || "加载失败，请重试");
    } finally {
      setLoading(false);
    }
  }, []);

  /** 加载更多（下一页） */
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
      const list = (res.list as unknown as TaskVO[]) || [];
      setTaskList((prev) => [...prev, ...list]);
      setPageNum(nextPage);
      setHasMore(list.length >= PAGE_SIZE);
    } catch (err: any) {
      Taro.showToast({ title: err?.message || "加载更多失败", icon: "none" });
    }
  }, [loading, hasMore, pageNum, statusFilter]);

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
    loadTaskList(statusFilter);
  });

  usePullDownRefresh(() => {
    loadTaskList(statusFilter).finally(() => {
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
      loadTaskList(status);
    },
    [stopPolling, loadTaskList]
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
  const handleCancel = useCallback((task: TaskVO) => {
    Taro.showModal({
      title: "取消任务",
      content: "确认取消该任务吗？取消后不可恢复。",
      confirmColor: "#ff4d4f",
      success: async (res) => {
        if (!res.confirm) return;
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
          setDetailTask((prev) =>
            prev?.taskId === task.taskId ? updated : prev
          );
          Taro.showToast({ title: "任务已取消", icon: "success" });
        } catch (err: any) {
          Taro.showToast({ title: err?.message || "取消失败", icon: "none" });
        } finally {
          setCancelLoadingId(null);
        }
      },
    });
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
    } catch (err: any) {
      Taro.showToast({ title: err?.message || "下载失败", icon: "none" });
    } finally {
      setDownloadLoadingId(null);
    }
  }, []);

  // ==================== 渲染 ====================

  /** 渲染任务卡片 */
  const renderTaskCard = (task: TaskVO) => {
    const tagInfo = STATUS_TAG[task.status] || {
      label: task.status,
      color: "#8c8c8c",
    };
    const isActive = POLLING_STATUSES.includes(task.status);
    const canDownload = task.status === "COMPLETED";

    return (
      <View
        key={task.taskId}
        className="task-card"
        onClick={() => handleDetail(task)}
      >
        <View className="card-header">
          <View className="header-left">
            <Tag color={tagInfo.color} size="small">
              {tagInfo.label}
            </Tag>
            <Text className="task-type">
              {TASK_TYPE_LABEL[task.taskType || ""] || task.taskType || "未知"}
            </Text>
          </View>
          <Text className="task-id">{shortTaskId(task.taskId)}</Text>
        </View>

        {isActive && (
          <View className="card-progress">
            <Progress percent={task.progress || 0} color="primary" />
          </View>
        )}

        {task.status === "FAILED" && task.error && (
          <View className="card-error">
            <Text>{task.error}</Text>
          </View>
        )}

        <View className="card-footer">
          <Text className="task-time">创建: {formatTime(task.createdAt)}</Text>
          <View className="task-actions" onClick={(e) => e.stopPropagation()}>
            {isActive && (
              <Button
                size="mini"
                color="danger"
                loading={cancelLoadingId === task.taskId}
                onClick={() => handleCancel(task)}
              >
                取消
              </Button>
            )}
            {canDownload && (
              <Button
                size="mini"
                color="primary"
                loading={downloadLoadingId === task.taskId}
                onClick={() => handleDownload(task)}
              >
                下载
              </Button>
            )}
          </View>
        </View>
      </View>
    );
  };

  /** 渲染详情弹窗中的描述项 */
  const renderDetailItem = (label: string, value: React.ReactNode) => (
    <View className="detail-item">
      <Text className="detail-label">{label}</Text>
      <View className="detail-value">{value}</View>
    </View>
  );

  return (
    <View className="task-page">
      <Navbar title="任务中心">
        <Navbar.NavLeft>
          <ArrowLeft />
        </Navbar.NavLeft>
      </Navbar>

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
            onRetry={() => loadTaskList(statusFilter)}
          />
        ) : taskList.length === 0 ? (
          <Empty>
            <Empty.Description>暂无任务</Empty.Description>
          </Empty>
        ) : (
          <>
            {taskList.map(renderTaskCard)}
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
      <Popup
        open={detailVisible}
        placement="bottom"
        rounded
        onClose={handleDetailClose}
        className="detail-popup"
      >
        {detailTask && (
          <View className="detail-content">
            <View className="detail-header">
              <Text className="detail-title">任务详情</Text>
              <Text className="detail-close" onClick={handleDetailClose}>
                关闭
              </Text>
            </View>

            {renderDetailItem("任务ID", detailTask.taskId)}
            {renderDetailItem(
              "任务类型",
              TASK_TYPE_LABEL[detailTask.taskType || ""] ||
                detailTask.taskType ||
                "-"
            )}
            {renderDetailItem(
              "状态",
              <Tag
                color={STATUS_TAG[detailTask.status]?.color || "default"}
                size="small"
              >
                {STATUS_TAG[detailTask.status]?.label || detailTask.status}
              </Tag>
            )}

            {POLLING_STATUSES.includes(detailTask.status) && (
              <View className="detail-progress">
                <Progress percent={detailTask.progress || 0} color="primary" />
              </View>
            )}

            {detailTask.totalFiles != null &&
              renderDetailItem(
                "文件处理",
                `${detailTask.processedFiles || 0} / ${detailTask.totalFiles}`
              )}
            {renderDetailItem("创建时间", formatTime(detailTask.createdAt))}
            {renderDetailItem("开始时间", formatTime(detailTask.startedAt))}
            {renderDetailItem("完成时间", formatTime(detailTask.completedAt))}
            {detailTask.expiresAt &&
              renderDetailItem("过期时间", formatTime(detailTask.expiresAt))}
            {detailTask.error && renderDetailItem("错误信息", detailTask.error)}

            {/* 详情弹窗操作按钮 */}
            <View className="detail-footer">
              {POLLING_STATUSES.includes(detailTask.status) && (
                <Button
                  block
                  color="danger"
                  loading={cancelLoadingId === detailTask.taskId}
                  onClick={() => handleCancel(detailTask)}
                >
                  取消任务
                </Button>
              )}
              {detailTask.status === "COMPLETED" && (
                <Button
                  block
                  color="primary"
                  loading={downloadLoadingId === detailTask.taskId}
                  onClick={() => handleDownload(detailTask)}
                >
                  下载结果
                </Button>
              )}
              {TERMINAL_STATUSES.includes(detailTask.status) &&
                detailTask.status !== "COMPLETED" && (
                  <Button block onClick={handleDetailClose}>
                    关闭
                  </Button>
                )}
            </View>
          </View>
        )}
      </Popup>
    </View>
  );
};

export default TaskPage;
