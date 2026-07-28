import {
  FeedbackAPI,
  type FeedbackCloseForm,
  type FeedbackPageVO,
  type FeedbackQuery,
  type FeedbackStatus,
  type FeedbackType,
} from "dehaze-sdk-js";
import { useHasPerm } from "@/hooks/usePermission";
import {
  Button,
  Card,
  DatePicker,
  Form,
  Input,
  InputNumber,
  message,
  Modal,
  Select,
  Space,
  Table,
  Tag,
  type TableColumnsType,
} from "antd";
import {
  BarChartOutlined,
  CloseCircleOutlined,
  EyeOutlined,
  MessageOutlined,
  ReloadOutlined,
  SearchOutlined,
  TagsOutlined,
  UserOutlined,
} from "@ant-design/icons";
import type { Dayjs } from "dayjs";
import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useNavigate } from "react-router-dom";
import FeedbackAssignDialog, {
  type FeedbackAssignDialogRef,
} from "./components/FeedbackAssignDialog";
import FeedbackDetailDrawer, {
  type FeedbackDetailDrawerRef,
} from "./components/FeedbackDetailDrawer";
import FeedbackReplyDialog, {
  type FeedbackReplyDialogRef,
} from "./components/FeedbackReplyDialog";
import FeedbackTagDialog, {
  type FeedbackTagDialogRef,
} from "./components/FeedbackTagDialog";
import "./index.scss";

const { RangePicker } = DatePicker;

const TYPE_OPTIONS: { value: FeedbackType; label: string }[] = [
  { value: "suggestion", label: "功能建议" },
  { value: "bug", label: "问题报告" },
  { value: "experience", label: "体验反馈" },
  { value: "complaint", label: "投诉" },
];

const STATUS_OPTIONS: { value: FeedbackStatus; label: string }[] = [
  { value: "pending", label: "待处理" },
  { value: "processing", label: "处理中" },
  { value: "replied", label: "已回复" },
  { value: "closed", label: "已关闭" },
];

const PRIORITY_OPTIONS: { value: number; label: string }[] = [
  { value: 1, label: "低" },
  { value: 2, label: "中" },
  { value: 3, label: "高" },
  { value: 4, label: "紧急" },
];

const TYPE_LABEL: Record<FeedbackType, string> = {
  suggestion: "功能建议",
  bug: "问题报告",
  experience: "体验反馈",
  complaint: "投诉",
};
const TYPE_COLOR: Record<FeedbackType, string> = {
  suggestion: "blue",
  bug: "red",
  experience: "green",
  complaint: "orange",
};
const STATUS_LABEL: Record<FeedbackStatus, string> = {
  pending: "待处理",
  processing: "处理中",
  replied: "已回复",
  closed: "已关闭",
};
const STATUS_COLOR: Record<FeedbackStatus, string> = {
  pending: "orange",
  processing: "blue",
  replied: "green",
  closed: "default",
};
const PRIORITY_LABEL: Record<number, string> = {
  1: "低",
  2: "中",
  3: "高",
  4: "紧急",
};
const PRIORITY_COLOR: Record<number, string> = {
  1: "default",
  2: "blue",
  3: "orange",
  4: "red",
};

const FeedbackManagement: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [pageData, setPageData] = useState<FeedbackPageVO[]>([]);
  const [total, setTotal] = useState(0);
  const [searchForm] = Form.useForm();
  const [queryParams, setQueryParams] = useState<FeedbackQuery>({
    pageNum: 1,
    pageSize: 10,
  });
  const [refreshFlag, setRefreshFlag] = useState(0);

  const detailDrawerRef = useRef<FeedbackDetailDrawerRef>(null);
  const assignDialogRef = useRef<FeedbackAssignDialogRef>(null);
  const replyDialogRef = useRef<FeedbackReplyDialogRef>(null);
  const tagDialogRef = useRef<FeedbackTagDialogRef>(null);

  const hasPerm = useHasPerm();
  const navigate = useNavigate();

  const loadData = useCallback(async (params: FeedbackQuery) => {
    setLoading(true);
    try {
      const result = await FeedbackAPI.listFeedback(params);
      setPageData(result.list || []);
      setTotal(result.total || 0);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData(queryParams);
  }, [queryParams, refreshFlag]);

  const refreshList = useCallback(() => {
    setRefreshFlag((prev) => prev + 1);
  }, []);

  const handleSearch = useCallback(
    (values: {
      keywords?: string;
      feedbackType?: FeedbackType;
      status?: FeedbackStatus;
      priority?: number;
      assigneeId?: number;
      timeRange?: [Dayjs, Dayjs];
    }) => {
      setQueryParams((prev) => ({
        ...prev,
        pageNum: 1,
        keywords: values.keywords || undefined,
        feedbackType: values.feedbackType,
        status: values.status,
        priority: values.priority,
        assigneeId: values.assigneeId,
        startTime: values.timeRange?.[0]?.format("YYYY-MM-DD 00:00:00"),
        endTime: values.timeRange?.[1]?.format("YYYY-MM-DD 23:59:59"),
      }));
    },
    []
  );

  const handleReset = useCallback(() => {
    searchForm.resetFields();
    setQueryParams({ pageNum: 1, pageSize: 10 });
  }, [searchForm]);

  const handlePageChange = useCallback((page: number, pageSize: number) => {
    setQueryParams((prev) => ({ ...prev, pageNum: page, pageSize }));
  }, []);

  const handleDetail = useCallback((record: FeedbackPageVO) => {
    detailDrawerRef.current?.open(record);
  }, []);

  const handleAssign = useCallback((record: FeedbackPageVO) => {
    assignDialogRef.current?.open(record.id, record.assigneeId);
  }, []);

  const handleReply = useCallback((record: FeedbackPageVO) => {
    replyDialogRef.current?.open(record.id);
  }, []);

  const handleEditTag = useCallback((record: FeedbackPageVO) => {
    tagDialogRef.current?.open(record.id, record.tags);
  }, []);

  const handleClose = useCallback(
    (record: FeedbackPageVO) => {
      let closeReason = "";
      Modal.confirm({
        title: "关闭反馈",
        content: (
          <Input.TextArea
            rows={4}
            maxLength={500}
            showCount
            placeholder="请输入关闭原因"
            onChange={(e) => {
              closeReason = e.target.value;
            }}
          />
        ),
        okText: "确定",
        cancelText: "取消",
        okType: "danger",
        onOk: () => {
          const reason = closeReason.trim();
          if (!reason) {
            message.error("关闭原因不能为空");
            return Promise.reject(new Error("empty"));
          }
          const form: FeedbackCloseForm = { closeReason: reason };
          return FeedbackAPI.closeFeedback(record.id, form)
            .then(() => {
              message.success("已关闭");
              refreshList();
            })
            .catch((error) => {
              message.error(error?.message || "关闭失败");
              return Promise.reject(error);
            });
        },
      });
    },
    [refreshList]
  );

  const handleGoStats = useCallback(() => {
    navigate("/feedback/stats?tab=feedback");
  }, [navigate]);

  const columns: TableColumnsType<FeedbackPageVO> = useMemo(
    () => [
      {
        title: "编号",
        dataIndex: "id",
        key: "id",
        width: 80,
        align: "center",
      },
      {
        title: "标题",
        dataIndex: "title",
        key: "title",
        minWidth: 200,
        ellipsis: true,
      },
      {
        title: "类型",
        dataIndex: "feedbackType",
        key: "feedbackType",
        width: 100,
        align: "center",
        render: (type: FeedbackType) => (
          <Tag color={TYPE_COLOR[type]}>{TYPE_LABEL[type]}</Tag>
        ),
      },
      {
        title: "模块",
        dataIndex: "relatedModule",
        key: "relatedModule",
        width: 120,
        align: "center",
        render: (mod?: string) => mod || "-",
      },
      {
        title: "状态",
        dataIndex: "status",
        key: "status",
        width: 100,
        align: "center",
        render: (status: FeedbackStatus) => (
          <Tag color={STATUS_COLOR[status]}>{STATUS_LABEL[status]}</Tag>
        ),
      },
      {
        title: "优先级",
        dataIndex: "priority",
        key: "priority",
        width: 90,
        align: "center",
        render: (priority: number) => (
          <Tag color={PRIORITY_COLOR[priority]}>
            {PRIORITY_LABEL[priority] || String(priority)}
          </Tag>
        ),
      },
      {
        title: "处理人",
        key: "assignee",
        width: 110,
        align: "center",
        render: (_: unknown, record: FeedbackPageVO) =>
          record.assigneeName || (
            <span style={{ color: "#909399" }}>未分配</span>
          ),
      },
      {
        title: "提交时间",
        dataIndex: "createTime",
        key: "createTime",
        width: 170,
        align: "center",
      },
      {
        title: "操作",
        key: "action",
        width: 340,
        align: "center",
        fixed: "right",
        render: (_: unknown, record: FeedbackPageVO) => (
          <Space size="small" wrap>
            <Button
              type="link"
              size="small"
              icon={<EyeOutlined />}
              onClick={() => handleDetail(record)}
            >
              详情
            </Button>
            {hasPerm("feedback:assign") && (
              <Button
                type="link"
                size="small"
                icon={<UserOutlined />}
                disabled={record.status === "closed"}
                onClick={() => handleAssign(record)}
              >
                分配
              </Button>
            )}
            {hasPerm("feedback:reply") && (
              <Button
                type="link"
                size="small"
                icon={<MessageOutlined />}
                disabled={record.status === "closed"}
                onClick={() => handleReply(record)}
              >
                回复
              </Button>
            )}
            {hasPerm("feedback:edit") && (
              <Button
                type="link"
                size="small"
                icon={<TagsOutlined />}
                onClick={() => handleEditTag(record)}
              >
                标签
              </Button>
            )}
            {hasPerm("feedback:close") && record.status !== "closed" && (
              <Button
                type="link"
                size="small"
                danger
                icon={<CloseCircleOutlined />}
                onClick={() => handleClose(record)}
              >
                关闭
              </Button>
            )}
          </Space>
        ),
      },
    ],
    [
      handleDetail,
      handleAssign,
      handleReply,
      handleEditTag,
      handleClose,
      hasPerm,
    ]
  );

  return (
    <div className="feedback-management-container">
      <Card className="search-card" size="small">
        <Form form={searchForm} layout="inline" onFinish={handleSearch}>
          <Form.Item name="keywords" label="关键字">
            <Input
              placeholder="标题/内容/用户名"
              allowClear
              style={{ width: 180 }}
            />
          </Form.Item>
          <Form.Item name="feedbackType" label="类型">
            <Select
              placeholder="全部"
              allowClear
              style={{ width: 140 }}
              options={TYPE_OPTIONS}
            />
          </Form.Item>
          <Form.Item name="status" label="状态">
            <Select
              placeholder="全部"
              allowClear
              style={{ width: 140 }}
              options={STATUS_OPTIONS}
            />
          </Form.Item>
          <Form.Item name="priority" label="优先级">
            <Select
              placeholder="全部"
              allowClear
              style={{ width: 140 }}
              options={PRIORITY_OPTIONS}
            />
          </Form.Item>
          <Form.Item name="assigneeId" label="处理人">
            <InputNumber
              min={1}
              placeholder="处理人ID"
              style={{ width: 140 }}
            />
          </Form.Item>
          <Form.Item name="timeRange" label="提交时间">
            <RangePicker style={{ width: 240 }} />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button
                type="primary"
                htmlType="submit"
                icon={<SearchOutlined />}
              >
                搜索
              </Button>
              <Button
                htmlType="reset"
                icon={<ReloadOutlined />}
                onClick={handleReset}
              >
                重置
              </Button>
              <Button
                type="link"
                icon={<BarChartOutlined />}
                onClick={handleGoStats}
              >
                反馈统计
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Card>

      <Card className="table-card" size="small">
        <Table
          columns={columns}
          dataSource={pageData}
          rowKey={(record) => record.id}
          loading={loading}
          scroll={{ x: 1300 }}
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

      <FeedbackDetailDrawer ref={detailDrawerRef} />
      <FeedbackAssignDialog ref={assignDialogRef} onSuccess={refreshList} />
      <FeedbackReplyDialog ref={replyDialogRef} onSuccess={refreshList} />
      <FeedbackTagDialog ref={tagDialogRef} onSuccess={refreshList} />
    </div>
  );
};

export default FeedbackManagement;
