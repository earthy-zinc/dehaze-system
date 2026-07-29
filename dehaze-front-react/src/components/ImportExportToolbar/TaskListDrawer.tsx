import {
  cancelTask,
  clearPollingTimer,
  fetchTaskList,
  fetchTaskStatus,
  POLLING_STATUSES,
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
  Drawer,
  Modal,
  Pagination,
  Progress,
  Radio,
  Select,
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
import type { TaskCategory, TaskStatus, TaskVO } from "dehaze-sdk-js";
import {
  STATUS_COLOR_MAP,
  STATUS_LABEL_MAP,
  TASK_TYPE_LABEL_MAP,
} from "./types";

const POLLING_INTERVAL = 3000;

interface TaskListDrawerProps {
  open: boolean;
  module?: string;
  onClose: () => void;
}

const STATUS_OPTIONS: { label: string; value: TaskStatus | "" }[] = [
  { label: "全部", value: "" },
  { label: "待执行", value: 1 },
  { label: "执行中", value: 2 },
  { label: "已完成", value: 3 },
  { label: "失败", value: 4 },
  { label: "已取消", value: 5 },
];

const formatDateTime = (value?: string): string => {
  if (!value) return "-";
  const date = new Date(value);
  if (isNaN(date.getTime())) return String(value);
  return date.toLocaleString("zh-CN");
};

const isImportTask = (taskType?: string): boolean => {
  if (!taskType) return false;
  return taskType.endsWith("_import");
};

const TaskListDrawer: React.FC<TaskListDrawerProps> = ({
  open,
  module,
  onClose,
}) => {
  const dispatch = useDispatch<DisPatchType>();
  const { taskList, total, loading } = useSelector(
    (state: RootState) => state.task
  );

  const [pageNum, setPageNum] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [statusFilter, setStatusFilter] = useState<TaskStatus | "">("");
  const [categoryFilter, setCategoryFilter] = useState<string>("");
  const [refreshFlag, setRefreshFlag] = useState(0);
  const [downloadLoadingId, setDownloadLoadingId] = useState<string | null>(
    null
  );
  const [cancelLoadingId, setCancelLoadingId] = useState<string | null>(null);

  const taskListRef = useRef<TaskVO[]>([]);
  useEffect(() => {
    taskListRef.current = taskList;
  }, [taskList]);

  const buildTaskTypeFilter = (): string | undefined => {
    if (module) {
      return [`${module}_export`, `${module}_import`].join(",");
    }
    return undefined;
  };

  const loadTaskList = useCallback(async () => {
    try {
      await dispatch(
        fetchTaskList({
          pageNum,
          pageSize,
          status: statusFilter || undefined,
          taskCategory: (categoryFilter || undefined) as
            TaskCategory | undefined,
          taskType: buildTaskTypeFilter(),
        })
      );
    } catch (error: unknown) {
      message.error((error as Error)?.message || "任务列表加载失败");
    }
  }, [dispatch, pageNum, pageSize, statusFilter, categoryFilter, module]);

  useEffect(() => {
    if (open) {
      loadTaskList();
    }
  }, [open, pageNum, pageSize, refreshFlag, loadTaskList]);

  const refreshList = useCallback(() => {
    setRefreshFlag((prev) => prev + 1);
  }, []);

  const pollingTimerRef = useRef<number | null>(null);
  const pollingTimer = useSelector(
    (state: RootState) => state.task.pollingTimer
  );
  useEffect(() => {
    pollingTimerRef.current = pollingTimer;
  }, [pollingTimer]);

  const stopPolling = useCallback(() => {
    if (pollingTimerRef.current) {
      window.clearInterval(pollingTimerRef.current);
      dispatch(clearPollingTimer());
    }
  }, [dispatch]);

  const pollTaskStatuses = useCallback(() => {
    const pollingTasks = taskListRef.current.filter((task) =>
      POLLING_STATUSES.includes(task.status as TaskStatus)
    );
    if (pollingTasks.length === 0) {
      stopPolling();
      return;
    }
    pollingTasks.forEach((task) => {
      dispatch(fetchTaskStatus(task.taskId));
    });
  }, [dispatch, stopPolling]);

  const startPolling = useCallback(() => {
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

  useEffect(() => {
    if (!open) return;
    const hasPollingTask = taskList.some((task) =>
      POLLING_STATUSES.includes(task.status as TaskStatus)
    );
    if (hasPollingTask) {
      startPolling();
    } else {
      stopPolling();
    }
  }, [taskList, open, startPolling, stopPolling]);

  useEffect(() => {
    if (!open) return;
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
  }, [open, startPolling, stopPolling]);

  useEffect(() => {
    if (!open && pollingTimerRef.current) {
      stopPolling();
    }
  }, [open, stopPolling]);

  useEffect(() => {
    return () => {
      if (pollingTimerRef.current) {
        window.clearInterval(pollingTimerRef.current);
      }
    };
  }, []);

  const handleFilterChange = () => {
    setPageNum(1);
    setRefreshFlag((prev) => prev + 1);
  };

  const handleCancelTask = useCallback(
    (record: TaskVO) => {
      Modal.confirm({
        title: "取消任务",
        content: "确认取消该任务吗？取消后不可恢复。",
        okText: "确定",
        cancelText: "取消",
        okType: "danger",
        onOk: () => {
          setCancelLoadingId(record.taskId);
          return dispatch(cancelTask(record.taskId))
            .unwrap()
            .then(() => {
              message.success("任务已取消");
              setCancelLoadingId(null);
              refreshList();
            })
            .catch((error: unknown) => {
              setCancelLoadingId(null);
              message.error((error as Error)?.message || "取消任务失败");
              return Promise.reject(error);
            });
        },
      });
    },
    [dispatch, refreshList]
  );

  const handleDownload = useCallback((record: TaskVO) => {
    if (record.status !== 3) {
      message.warning("任务尚未完成，无法下载");
      return;
    }
    if (!record.downloadUrl) {
      message.warning("下载链接不存在或已过期");
      return;
    }
    if (record.expiresAt) {
      const expiresAt = new Date(record.expiresAt).getTime();
      if (Date.now() > expiresAt) {
        message.warning("任务结果已过期，无法下载");
        return;
      }
    }
    setDownloadLoadingId(record.taskId);
    try {
      const link = document.createElement("a");
      link.href = record.downloadUrl;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      message.success("开始下载");
    } finally {
      setDownloadLoadingId(null);
    }
  }, []);

  const columns: TableColumnsType<TaskVO> = useMemo(
    () => [
      {
        title: "任务ID",
        dataIndex: "taskId",
        key: "taskId",
        width: 220,
        ellipsis: true,
      },
      {
        title: "类型",
        dataIndex: "taskType",
        key: "taskType",
        width: 120,
        align: "center",
        render: (taskType: string) =>
          TASK_TYPE_LABEL_MAP[taskType] || taskType || "-",
      },
      {
        title: "状态",
        dataIndex: "status",
        key: "status",
        width: 100,
        align: "center",
        render: (status: TaskStatus) => (
          <Tag color={STATUS_COLOR_MAP[status] || "default"}>
            {STATUS_LABEL_MAP[status] || status}
          </Tag>
        ),
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
          if (status === 3) {
            progressStatus = "success";
          } else if (status === 4) {
            progressStatus = "exception";
          } else if (status === 5) {
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
        title: "操作",
        key: "action",
        width: 200,
        align: "center",
        fixed: "right",
        render: (_: unknown, record: TaskVO) => (
          <Space size="small">
            {(record.status === 1 || record.status === 2) && (
              <Button
                type="link"
                size="small"
                danger
                icon={<CloseCircleOutlined />}
                loading={cancelLoadingId === record.taskId}
                onClick={() => handleCancelTask(record)}
              >
                取消
              </Button>
            )}
            {record.status === 3 && isImportTask(record.taskType) && (
              <Button
                type="link"
                size="small"
                icon={<EyeOutlined />}
                loading={downloadLoadingId === record.taskId}
                onClick={() => handleDownload(record)}
              >
                查看结果
              </Button>
            )}
            {record.status === 3 && !isImportTask(record.taskType) && (
              <Button
                type="link"
                size="small"
                icon={<DownloadOutlined />}
                loading={downloadLoadingId === record.taskId}
                onClick={() => handleDownload(record)}
              >
                下载
              </Button>
            )}
          </Space>
        ),
      },
    ],
    [cancelLoadingId, downloadLoadingId, handleCancelTask, handleDownload]
  );

  const drawerTitle = module
    ? `${TASK_TYPE_LABEL_MAP[`${module}_export`]?.replace("导出", "") || ""}任务列表`
    : "任务列表";

  return (
    <Drawer
      title={drawerTitle}
      open={open}
      onClose={onClose}
      width={960}
      destroyOnHidden
    >
      <Space direction="vertical" size="middle" style={{ width: "100%" }}>
        <Space wrap>
          <Radio.Group
            buttonStyle="solid"
            value={categoryFilter}
            onChange={(e) => {
              setCategoryFilter(e.target.value);
              handleFilterChange();
            }}
          >
            <Radio.Button value="">全部</Radio.Button>
            <Radio.Button value="import">导入</Radio.Button>
            <Radio.Button value="export">导出</Radio.Button>
          </Radio.Group>
          <Select
            value={statusFilter}
            style={{ width: 140 }}
            onChange={(value) => {
              setStatusFilter(value);
              handleFilterChange();
            }}
            options={STATUS_OPTIONS}
            placeholder="任务状态"
          />
          <Button icon={<ReloadOutlined />} onClick={refreshList}>
            刷新
          </Button>
        </Space>

        <Table
          columns={columns}
          dataSource={taskList}
          rowKey={(record) => record.taskId}
          loading={loading}
          scroll={{ x: 1100 }}
          pagination={false}
        />

        <Pagination
          current={pageNum}
          pageSize={pageSize}
          total={total}
          showSizeChanger
          showQuickJumper
          pageSizeOptions={["10", "20", "50", "100"]}
          showTotal={(t) => `共 ${t} 条`}
          onChange={(page, size) => {
            setPageNum(page);
            setPageSize(size);
          }}
        />
      </Space>
    </Drawer>
  );
};

export default TaskListDrawer;
