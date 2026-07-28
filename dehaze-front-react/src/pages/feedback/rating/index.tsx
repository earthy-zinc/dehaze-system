import {
  AlgorithmAPI,
  FeedbackAPI,
  type Algorithm,
  type RatingPageVO,
  type RatingQuery,
} from "dehaze-sdk-js";
import { useHasPerm } from "@/hooks/usePermission";
import {
  Button,
  Card,
  DatePicker,
  Form,
  Input,
  message,
  Modal,
  Rate,
  Select,
  Space,
  Table,
  Tag,
  type TableColumnsType,
} from "antd";
import {
  BarChartOutlined,
  EyeInvisibleOutlined,
  EyeOutlined,
  MessageOutlined,
  ReloadOutlined,
  SearchOutlined,
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
import RatingDetailDrawer, {
  type RatingDetailDrawerRef,
} from "./components/RatingDetailDrawer";
import RatingReplyDialog, {
  type RatingReplyDialogRef,
} from "./components/RatingReplyDialog";
import "./index.scss";

const { RangePicker } = DatePicker;

const RatingManagement: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [pageData, setPageData] = useState<RatingPageVO[]>([]);
  const [total, setTotal] = useState(0);
  const [searchForm] = Form.useForm();
  const [algorithmOptions, setAlgorithmOptions] = useState<
    { value: number; label: string }[]
  >([]);
  const [queryParams, setQueryParams] = useState<RatingQuery>({
    pageNum: 1,
    pageSize: 10,
  });
  const [refreshFlag, setRefreshFlag] = useState(0);

  const detailDrawerRef = useRef<RatingDetailDrawerRef>(null);
  const replyDialogRef = useRef<RatingReplyDialogRef>(null);

  const hasPerm = useHasPerm();
  const navigate = useNavigate();

  const loadData = useCallback(async (params: RatingQuery) => {
    setLoading(true);
    try {
      const result = await FeedbackAPI.listRatings(params);
      setPageData(result.list || []);
      setTotal(result.total || 0);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData(queryParams);
  }, [queryParams, refreshFlag]);

  useEffect(() => {
    AlgorithmAPI.listAll().then((list: Algorithm[]) => {
      setAlgorithmOptions(list.map((a) => ({ value: a.id, label: a.name })));
    });
  }, []);

  const refreshList = useCallback(() => {
    setRefreshFlag((prev) => prev + 1);
  }, []);

  const handleSearch = useCallback(
    (values: {
      keywords?: string;
      algorithmId?: number;
      ratingMin?: number;
      ratingMax?: number;
      hasComment?: boolean;
      timeRange?: [Dayjs, Dayjs];
    }) => {
      const next: RatingQuery = {
        ...queryParams,
        pageNum: 1,
        keywords: values.keywords || undefined,
        algorithmId: values.algorithmId,
        ratingMin: values.ratingMin,
        ratingMax: values.ratingMax,
        hasComment: values.hasComment,
        startTime: values.timeRange?.[0]?.format("YYYY-MM-DD 00:00:00"),
        endTime: values.timeRange?.[1]?.format("YYYY-MM-DD 23:59:59"),
      };
      setQueryParams(next);
    },
    [queryParams]
  );

  const handleReset = useCallback(() => {
    searchForm.resetFields();
    setQueryParams({ pageNum: 1, pageSize: 10 });
  }, [searchForm]);

  const handlePageChange = useCallback((page: number, pageSize: number) => {
    setQueryParams((prev) => ({ ...prev, pageNum: page, pageSize }));
  }, []);

  const handleDetail = useCallback((record: RatingPageVO) => {
    detailDrawerRef.current?.open(record);
  }, []);

  const handleHide = useCallback(
    (record: RatingPageVO) => {
      const action = record.isHidden === 1 ? "显示" : "隐藏";
      Modal.confirm({
        title: "提示",
        content: `确认${action}该条评价吗？`,
        okText: "确定",
        cancelText: "取消",
        onOk: () =>
          FeedbackAPI.hideRating(record.id)
            .then(() => {
              message.success(`${action}成功`);
              refreshList();
            })
            .catch((error) => {
              message.error(error?.message || `${action}失败`);
              return Promise.reject(error);
            }),
      });
    },
    [refreshList]
  );

  const handleReply = useCallback((record: RatingPageVO) => {
    replyDialogRef.current?.open(record.id);
  }, []);

  const handleGoStats = useCallback(() => {
    navigate("/feedback/stats?tab=rating");
  }, [navigate]);

  const columns: TableColumnsType<RatingPageVO> = useMemo(
    () => [
      {
        title: "用户名",
        key: "username",
        width: 120,
        align: "center",
        render: (_: unknown, record: RatingPageVO) =>
          record.isAnonymous === 1 ? (
            <span style={{ color: "#909399" }}>匿名用户</span>
          ) : (
            record.username || "-"
          ),
      },
      {
        title: "算法",
        dataIndex: "algorithmName",
        key: "algorithmName",
        width: 120,
        align: "center",
      },
      {
        title: "评分",
        dataIndex: "rating",
        key: "rating",
        width: 140,
        align: "center",
        render: (rating: number) => <Rate disabled value={rating} />,
      },
      {
        title: "评价内容",
        dataIndex: "comment",
        key: "comment",
        minWidth: 200,
        ellipsis: true,
        render: (comment?: string) => comment || "-",
      },
      {
        title: "标签",
        key: "tags",
        width: 220,
        render: (_, record: RatingPageVO) => {
          if (!record.tags?.length) return "-";
          const visibleTags = record.tags.slice(0, 3);
          const rest = record.tags.length - visibleTags.length;
          return (
            <Space wrap size={[4, 4]}>
              {visibleTags.map((tag) => (
                <Tag key={tag} style={{ margin: 0 }}>
                  {tag}
                </Tag>
              ))}
              {rest > 0 && <Tag>+{rest}</Tag>}
            </Space>
          );
        },
      },
      {
        title: "评价时间",
        dataIndex: "createTime",
        key: "createTime",
        width: 170,
        align: "center",
      },
      {
        title: "操作",
        key: "action",
        width: 220,
        align: "center",
        fixed: "right",
        render: (_: unknown, record: RatingPageVO) => (
          <Space size="small">
            <Button
              type="link"
              size="small"
              icon={<EyeOutlined />}
              onClick={() => handleDetail(record)}
            >
              详情
            </Button>
            {hasPerm("feedback:rating:edit") && (
              <Button
                type="link"
                size="small"
                danger={record.isHidden !== 1}
                icon={<EyeInvisibleOutlined />}
                onClick={() => handleHide(record)}
              >
                {record.isHidden === 1 ? "显示" : "隐藏"}
              </Button>
            )}
            {hasPerm("feedback:rating:reply") && (
              <Button
                type="link"
                size="small"
                icon={<MessageOutlined />}
                onClick={() => handleReply(record)}
              >
                回复
              </Button>
            )}
          </Space>
        ),
      },
    ],
    [handleDetail, handleHide, handleReply, hasPerm]
  );

  return (
    <div className="rating-management-container">
      <Card className="search-card" size="small">
        <Form form={searchForm} layout="inline" onFinish={handleSearch}>
          <Form.Item name="keywords" label="关键字">
            <Input
              placeholder="用户名/评价内容"
              allowClear
              style={{ width: 180 }}
            />
          </Form.Item>
          <Form.Item name="algorithmId" label="算法">
            <Select
              placeholder="全部"
              allowClear
              style={{ width: 160 }}
              options={algorithmOptions}
            />
          </Form.Item>
          <Form.Item name="ratingMin" label="最低分">
            <Select
              placeholder="最低"
              allowClear
              style={{ width: 90 }}
              options={[1, 2, 3, 4, 5].map((n) => ({
                value: n,
                label: `${n}星`,
              }))}
            />
          </Form.Item>
          <Form.Item name="ratingMax" label="最高分">
            <Select
              placeholder="最高"
              allowClear
              style={{ width: 90 }}
              options={[1, 2, 3, 4, 5].map((n) => ({
                value: n,
                label: `${n}星`,
              }))}
            />
          </Form.Item>
          <Form.Item name="hasComment" label="有无评论">
            <Select
              placeholder="全部"
              allowClear
              style={{ width: 120 }}
              options={[
                { value: true, label: "有评论" },
                { value: false, label: "无评论" },
              ]}
            />
          </Form.Item>
          <Form.Item name="timeRange" label="时间范围">
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
                统计
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
          scroll={{ x: 1100 }}
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

      <RatingDetailDrawer ref={detailDrawerRef} />
      <RatingReplyDialog ref={replyDialogRef} onSuccess={refreshList} />
    </div>
  );
};

export default RatingManagement;
