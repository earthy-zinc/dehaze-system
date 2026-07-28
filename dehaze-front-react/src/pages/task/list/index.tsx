import {
  cancelTask,
  clearPollingTimer,
  fetchTaskList,
  fetchTaskStatus,
  POLLING_STATUSES,
  setCurrentTask,
  setPollingTimer,
} from "@/store/modules/taskSlice";
import { DisPatchType, RootState } from "@/store";
import {
  CloseCircleOutlined,
  DownloadOutlined,
  EyeOutlined,
  ReloadOutlined,
} from "@ant-design/icons";
import {
  Button,
  Card,
  Descriptions,
  Modal,
  Progress,
  Radio,
  Space,
  Table,
  Tag,
  message,
  type TableColumnsType,
} from "antd";
import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  type TaskVO,
  type TaskQuery,
  type TaskStatus,
  type TaskCategory,
} from "dehaze-sdk-js";
import "./index.scss";

/** 轮询间隔（毫秒） */
const POLLING_INTERVAL = 3000;

/** 任务状态映射 */
const STATUS_MAP: Record<string, { label: string; color: string }> = {
  PENDING: { label: "待执行", color: "blue" },
  PROCESSING: { label: "执行中", color: "blue" },
  COMPLETED: { label: "已完成", color: "green" },
  FAILED: { label: "失败", color: "red" },
  CANCELLED: { label: "已取消", color: "default" },
};

/** 任务类型映射 */
const TASK_TYPE_MAP: Record<string, string> = {
  dataset_export: "数据集导出",
  user_export: "用户导出",
  role_export: "角色导出",
  dept_export: "部门导出",
  menu_export: "菜单导出",
  dict_export: "字典导出",
  algorithm_export: "算法导出",
  user_import: "用户导入",
  role_import: "角色导入",
  dept_import: "部门导入",
  menu_import: "菜单导入",
  dict_import: "字典导入",
  algorithm_import: "算法导入",
};

/** 状态筛选选项 */
const STATUS_OPTIONS = [
  { label: "全部", value: "" },
  { label: "待执行", value: "PENDING" },
  { label: "执行中", value: "PROCESSING" },
  { label: "已完成", value: "COMPLETED" },
  { label: "失败", value: "FAILED" },
  { label: "已取消", value: "CANCELLED" },
];

/** 类别筛选选项 */
const CATEGORY_OPTIONS = [
  { label: "全部", value: "" },
  { label: "导入", value: "import" },
  { label: "导出", value: "export" },
];

/** 判断是否为导入任务 */
const isImportTask = (taskType?: string): boolean => {
  if (!taskType) return false;
  return taskType.endsWith("_import");
};

/** 格式化日期时间 */
function formatDateTime(value?: Date | string): string {
  if (!value) return "-";
  const date = typeof value === "string" ? new Date(value) : value;
  if (isNaN(date.getTime())) return String(value);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(
    date.getDate()
  )} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(
    date.getSeconds()
  )}`;
}

const TaskManagement: React.FC = () => {
  const dispatch = useDispatch<DisPatchType>();
  const { taskList, total, loading, currentTask, pollingTimer } = useSelector(
    (state: RootState) => state.task
  );

  // 查询参数
  const [queryParams, setQueryParams] = useState<TaskQuery>({
    pageNum: 1,
    pageSize: 10,
    status: undefined,
    taskCategory: undefined,
  });
  const [refreshFlag, setRefreshFlag] = useState(0);

  // 详情弹窗
  const [detailVisible, setDetailVisible] = useState(false);

  // 使用 ref 保存最新的任务列表，供轮询回调读取
  const taskListRef = useRef<TaskVO[]>([]);
  useEffect(() => {
    taskListRef.current = taskList;
  }, [taskList]);

  // 保存最新的 pollingTimer，便于清理函数读取
  const pollingTimerRef = useRef<number | null>(null);
  useEffect(() => {
    pollingTimerRef.current = pollingTimer;
  }, [pollingTimer]);

  // ==================== 数据加载 ====================

  /** 加载任务列表 */
  const loadTaskList = useCallback(
    async (params: TaskQuery) => {
      try {
        await dispatch(fetchTaskList(params));
      } catch (error: any) {
        message.error(error?.message || "任务列表加载失败");
      }
    },
    [dispatch]
  );

  useEffect(() => {
    loadTaskList(queryParams);
  }, [queryParams, refreshFlag]);

  const refreshList = useCallback(() => {
    setRefreshFlag((prev) => prev + 1);
  }, []);

  // ==================== 轮询逻辑 ====================

  /** 停止轮询 */
  const stopPolling = useCallback(() => {
    if (pollingTimerRef.current) {
      window.clearInterval(pollingTimerRef.current);
      dispatch(clearPollingTimer());
    }
  }, [dispatch]);

  /** 执行一次轮询：查询所有 pending/processing 任务状态 */
  const pollTaskStatuses = useCallback(() => {
    const pollingTasks = taskListRef.current.filter((task) =>
      POLLING_STATUSES.includes(task.status as TaskStatus)
    );
    // 无需轮询的任务，停止全局轮询
    if (pollingTasks.length === 0) {
      stopPolling();
      return;
    }
    pollingTasks.forEach((task) => {
      dispatch(fetchTaskStatus(task.taskId));
    });
  }, [dispatch, stopPolling]);

  /** 启动轮询 */
  const startPolling = useCallback(() => {
    // 已有定时器则不重复启动
    if (pollingTimerRef.current) return;
    const hasPollingTask = taskListRef.current.some((task) =>
      POLLING_STATUSES.includes(task.status as TaskStatus)
    );
    if (!hasPollingTask) return;
    const timer = window.setInterval(() => {
      pollTaskStatuses();
    }, POLLING_INTERVAL);
    dispatch(setPollingTimer(timer));
  }, [dispatch, pollTaskStatuses]);

  // 任务列表变化后，检查是否需要启动/停止轮询
  useEffect(() => {
    const hasPollingTask = taskList.some((task) =>
      POLLING_STATUSES.includes(task.status as TaskStatus)
    );
    if (hasPollingTask && !pollingTimer) {
      startPolling();
    } else if (!hasPollingTask && pollingTimer) {
      stopPolling();
    }
  }, [taskList]);

  // 页面可见性变化：隐藏时暂停轮询，可见时恢复
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.hidden) {
        stopPolling();
      } else {
        startPolling();
      }
    };
    document.addEventListener("visibilitychange", handleVisibilityChange);
    return () => {
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [startPolling, stopPolling]);

  // 组件卸载时清除定时器
  useEffect(() => {
    return () => {
      if (pollingTimerRef.current) {
        window.clearInterval(pollingTimerRef.current);
      }
    };
  }, []);

  // ==================== 事件处理 ====================

  /** 状态筛选变化 */
  const handleStatusChange = useCallback((value: string) => {
    setQueryParams((prev) => ({
      ...prev,
      pageNum: 1,
      status: (value || undefined) as TaskStatus | undefined,
    }));
  }, []);

  /** 类别筛选变化 */
  const handleCategoryChange = useCallback((value: string) => {
    setQueryParams((prev) => ({
      ...prev,
      pageNum: 1,
      taskCategory: (value || undefined) as TaskCategory | undefined,
    }));
  }, []);

  /** 分页变化 */
  const handlePageChange = useCallback((page: number, pageSize: number) => {
    setQueryParams((prev) => ({
      ...prev,
      pageNum: page,
      pageSize,
    }));
  }, []);

  /** 查看任务详情 */
  const handleViewDetail = useCallback(
    (record: TaskVO) => {
      dispatch(setCurrentTask(record));
      setDetailVisible(true);
    },
    [dispatch]
  );

  /** 关闭详情弹窗 */
  const handleDetailClose = useCallback(() => {
    setDetailVisible(false);
    dispatch(setCurrentTask(null));
  }, [dispatch]);

  /** 取消任务 */
  const handleCancelTask = useCallback(
    (record: TaskVO) => {
      Modal.confirm({
        title: "取消任务",
        content: "确认取消该任务吗？取消后不可恢复。",
        okText: "确定",
        cancelText: "取消",
        okType: "danger",
        onOk: () => {
          return dispatch(cancelTask(record.taskId))
            .unwrap()
            .then(() => {
              message.success("任务已取消");
            })
            .catch((error: any) => {
              message.error(error?.message || "取消任务失败");
              return Promise.reject(error);
            });
        },
      });
    },
    [dispatch]
  );

  /** 下载任务结果 */
  const handleDownload = useCallback((record: TaskVO) => {
    if (record.status !== "COMPLETED") {
      message.warning("任务尚未完成，无法下载");
      return;
    }
    if (!record.downloadUrl) {
      message.warning("下载链接不存在或已过期");
      return;
    }
    // 检查是否过期
    if (record.expiresAt) {
      const expiresAt = new Date(record.expiresAt).getTime();
      if (Date.now() > expiresAt) {
        message.warning("任务结果已过期，无法下载");
        return;
      }
    }
    // 触发下载
    const link = document.createElement("a");
    link.href = record.downloadUrl;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }, []);

  // ==================== 表格列定义 ====================

  const columns: TableColumnsType<TaskVO> = useMemo(
    () => [
      {
        title: "任务ID",
        dataIndex: "taskId",
        key: "taskId",
        width: 200,
        align: "center",
        ellipsis: true,
      },
      {
        title: "类型",
        dataIndex: "taskType",
        key: "taskType",
        width: 120,
        align: "center",
        render: (taskType: string) =>
          TASK_TYPE_MAP[taskType] || taskType || "-",
      },
      {
        title: "状态",
        dataIndex: "status",
        key: "status",
        width: 100,
        align: "center",
        render: (status: string) => {
          const info = STATUS_MAP[status] || {
            label: "未知",
            color: "default",
          };
          return <Tag color={info.color}>{info.label}</Tag>;
        },
      },
      {
        title: "进度",
        dataIndex: "progress",
        key: "progress",
        width: 200,
        align: "center",
        render: (progress: number, record: TaskVO) => {
          const status = record.status;
          let progressStatus: "active" | "success" | "exception" | "normal" =
            "active";
          if (status === "COMPLETED") {
            progressStatus = "success";
          } else if (status === "FAILED") {
            progressStatus = "exception";
          } else if (status === "CANCELLED") {
            progressStatus = "normal";
          }
          return (
            <Progress
              percent={progress || 0}
              size="small"
              status={progressStatus}
            />
          );
        },
      },
      {
        title: "创建时间",
        dataIndex: "createdAt",
        key: "createdAt",
        width: 180,
        align: "center",
        render: (text: string) => formatDateTime(text),
      },
      {
        title: "完成时间",
        dataIndex: "completedAt",
        key: "completedAt",
        width: 180,
        align: "center",
        render: (text: string) => (text ? formatDateTime(text) : "-"),
      },
      {
        title: "操作",
        key: "action",
        width: 240,
        align: "center",
        fixed: "right",
        render: (_: unknown, record: TaskVO) => (
          <Space size="small">
            <Button
              type="link"
              size="small"
              icon={<EyeOutlined />}
              onClick={() => handleViewDetail(record)}
            >
              详情
            </Button>
            {(record.status === "PENDING" ||
              record.status === "PROCESSING") && (
              <Button
                type="link"
                size="small"
                danger
                icon={<CloseCircleOutlined />}
                onClick={() => handleCancelTask(record)}
              >
                取消
              </Button>
            )}
            {record.status === "COMPLETED" && isImportTask(record.taskType) && (
              <Button
                type="link"
                size="small"
                icon={<DownloadOutlined />}
                onClick={() => handleDownload(record)}
              >
                查看结果
              </Button>
            )}
            {record.status === "COMPLETED" &&
              !isImportTask(record.taskType) && (
                <Button
                  type="link"
                  size="small"
                  icon={<DownloadOutlined />}
                  onClick={() => handleDownload(record)}
                >
                  下载
                </Button>
              )}
          </Space>
        ),
      },
    ],
    [handleViewDetail, handleCancelTask, handleDownload]
  );

  // ==================== 渲染 ====================

  return (
    <div className="task-management-container">
      {/* 状态筛选区域 */}
      <Card className="filter-card" size="small">
        <Space wrap>
          <span>任务类别：</span>
          <Radio.Group
            buttonStyle="solid"
            value={queryParams.taskCategory || ""}
            onChange={(e) => handleCategoryChange(e.target.value)}
          >
            {CATEGORY_OPTIONS.map((opt) => (
              <Radio.Button key={opt.value} value={opt.value}>
                {opt.label}
              </Radio.Button>
            ))}
          </Radio.Group>
          <span>状态筛选：</span>
          <Radio.Group
            buttonStyle="solid"
            value={queryParams.status || ""}
            onChange={(e) => handleStatusChange(e.target.value)}
          >
            {STATUS_OPTIONS.map((opt) => (
              <Radio.Button key={opt.value} value={opt.value}>
                {opt.label}
              </Radio.Button>
            ))}
          </Radio.Group>
          <Button icon={<ReloadOutlined />} onClick={refreshList}>
            刷新
          </Button>
        </Space>
      </Card>

      {/* 任务列表表格 */}
      <Card className="table-card" size="small">
        <Table
          columns={columns}
          dataSource={taskList}
          rowKey={(record) => record.taskId}
          loading={loading}
          scroll={{ x: 1200 }}
          pagination={{
            current: queryParams.pageNum,
            pageSize: queryParams.pageSize,
            total,
            showSizeChanger: true,
            showQuickJumper: true,
            pageSizeOptions: ["10", "20", "50", "100"],
            showTotal: (t) => `共 ${t} 条`,
            onChange: handlePageChange,
          }}
        />
      </Card>

      {/* 任务详情弹窗 */}
      <Modal
        title="任务详情"
        open={detailVisible}
        onCancel={handleDetailClose}
        width={640}
        footer={
          currentTask?.status === "COMPLETED" ? (
            <Space>
              <Button onClick={handleDetailClose}>关闭</Button>
              <Button
                type="primary"
                icon={<DownloadOutlined />}
                onClick={() => currentTask && handleDownload(currentTask)}
              >
                {isImportTask(currentTask?.taskType) ? "查看结果" : "下载结果"}
              </Button>
            </Space>
          ) : currentTask?.status === "PENDING" ||
            currentTask?.status === "PROCESSING" ? (
            <Space>
              <Button onClick={handleDetailClose}>关闭</Button>
              <Button
                danger
                icon={<CloseCircleOutlined />}
                onClick={() => {
                  if (currentTask) {
                    handleCancelTask(currentTask);
                    handleDetailClose();
                  }
                }}
              >
                取消任务
              </Button>
            </Space>
          ) : (
            <Button onClick={handleDetailClose}>关闭</Button>
          )
        }
      >
        {currentTask && (
          <Descriptions column={2} bordered size="small">
            <Descriptions.Item label="任务ID" span={2}>
              {currentTask.taskId}
            </Descriptions.Item>
            <Descriptions.Item label="任务类型">
              {TASK_TYPE_MAP[currentTask.taskType || ""] ||
                currentTask.taskType ||
                "-"}
            </Descriptions.Item>
            <Descriptions.Item label="状态">
              {(() => {
                const info = STATUS_MAP[currentTask.status] || {
                  label: "未知",
                  color: "default",
                };
                return <Tag color={info.color}>{info.label}</Tag>;
              })()}
            </Descriptions.Item>
            <Descriptions.Item label="进度" span={2}>
              <Progress
                percent={currentTask.progress || 0}
                status={
                  currentTask.status === "COMPLETED"
                    ? "success"
                    : currentTask.status === "FAILED"
                      ? "exception"
                      : currentTask.status === "CANCELLED"
                        ? "normal"
                        : "active"
                }
              />
            </Descriptions.Item>
            <Descriptions.Item label="文件数">
              {currentTask.totalFiles != null
                ? `${currentTask.processedFiles ?? 0} / ${currentTask.totalFiles}`
                : "-"}
            </Descriptions.Item>
            <Descriptions.Item label="创建时间">
              {formatDateTime(currentTask.createdAt)}
            </Descriptions.Item>
            <Descriptions.Item label="开始时间">
              {formatDateTime(currentTask.startedAt)}
            </Descriptions.Item>
            <Descriptions.Item label="完成时间">
              {formatDateTime(currentTask.completedAt)}
            </Descriptions.Item>
            <Descriptions.Item label="过期时间" span={2}>
              {formatDateTime(currentTask.expiresAt)}
            </Descriptions.Item>
            {currentTask.error && (
              <Descriptions.Item label="错误信息" span={2}>
                <span style={{ color: "#ff4d4f" }}>{currentTask.error}</span>
              </Descriptions.Item>
            )}
          </Descriptions>
        )}
      </Modal>
    </div>
  );
};

export default TaskManagement;
