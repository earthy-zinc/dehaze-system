import {
  AnnouncementAPI,
  type AnnouncementQuery,
  type AnnouncementVO,
} from "dehaze-sdk-js";
import { useHasPerm } from "@/hooks/usePermission";
import {
  DeleteOutlined,
  EditOutlined,
  PlusOutlined,
  ReloadOutlined,
  SearchOutlined,
} from "@ant-design/icons";
import {
  Button,
  Card,
  Form,
  Input,
  Modal,
  Popconfirm,
  Select,
  Space,
  Table,
  Tag,
  type TableColumnsType,
  message,
} from "antd";
import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import AnnouncementFormDialog, {
  type AnnouncementFormDialogRef,
} from "./components/AnnouncementFormDialog";
import "./index.scss";

const TYPE_OPTIONS = [
  { value: "maintenance", label: "系统维护" },
  { value: "feature", label: "功能更新" },
  { value: "activity", label: "活动通知" },
  { value: "operation", label: "运营公告" },
];

const TYPE_LABEL: Record<string, string> = {
  maintenance: "系统维护",
  feature: "功能更新",
  activity: "活动通知",
  operation: "运营公告",
};

const STATUS_OPTIONS = [
  { value: 1, label: "草稿" },
  { value: 2, label: "待发送" },
  { value: 3, label: "已发送" },
  { value: 4, label: "已取消" },
];

const STATUS_TAG: Record<number, { label: string; color: string }> = {
  1: { label: "草稿", color: "default" },
  2: { label: "待发送", color: "orange" },
  3: { label: "已发送", color: "green" },
  4: { label: "已取消", color: "default" },
};

const AnnouncementManagement: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [pageData, setPageData] = useState<AnnouncementVO[]>([]);
  const [total, setTotal] = useState(0);
  const [searchForm] = Form.useForm();
  const [queryParams, setQueryParams] = useState<AnnouncementQuery>({
    pageNum: 1,
    pageSize: 10,
  });
  const [refreshFlag, setRefreshFlag] = useState(0);

  const formDialogRef = useRef<AnnouncementFormDialogRef>(null);
  const hasPerm = useHasPerm();

  const loadData = useCallback((params: AnnouncementQuery) => {
    setLoading(true);
    AnnouncementAPI.getPage(params)
      .then((result) => {
        setPageData(result.list || []);
        setTotal(result.total || 0);
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    loadData(queryParams);
  }, [queryParams, refreshFlag]);

  const refreshList = useCallback(() => {
    setRefreshFlag((prev) => prev + 1);
  }, []);

  const handleSearch = useCallback(
    (values: { title?: string; type?: string; status?: number }) => {
      setQueryParams((prev) => ({
        ...prev,
        pageNum: 1,
        title: values.title || undefined,
        type: values.type || undefined,
        status: values.status,
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

  const handleAdd = useCallback(() => {
    formDialogRef.current?.open("add");
  }, []);

  const handleEdit = useCallback((record: AnnouncementVO) => {
    formDialogRef.current?.open("edit", record.id);
  }, []);

  const handleDelete = useCallback(
    (record: AnnouncementVO) => {
      AnnouncementAPI.deleteById(record.id)
        .then(() => {
          message.success(`公告「${record.title}」删除成功`);
          refreshList();
        })
        .catch((err) => message.error(err?.message || "删除失败"));
    },
    [refreshList]
  );

  const handleSend = useCallback(
    (record: AnnouncementVO) => {
      Modal.confirm({
        title: "提示",
        content: `确定立即发送公告「${record.title}」吗？`,
        okText: "确定",
        cancelText: "取消",
        onOk: () =>
          AnnouncementAPI.send(record.id)
            .then((res) => {
              message.success(`发送成功，共送达 ${res.sentCount} 位用户`);
              refreshList();
            })
            .catch((err) => {
              message.error(err?.message || "发送失败");
              return Promise.reject(err);
            }),
      });
    },
    [refreshList]
  );

  const handleCancel = useCallback(
    (record: AnnouncementVO) => {
      Modal.confirm({
        title: "提示",
        content: `确定取消定时公告「${record.title}」吗？`,
        okText: "确定",
        cancelText: "取消",
        onOk: () =>
          AnnouncementAPI.cancel(record.id)
            .then(() => {
              message.success("取消成功");
              refreshList();
            })
            .catch((err) => {
              message.error(err?.message || "取消失败");
              return Promise.reject(err);
            }),
      });
    },
    [refreshList]
  );

  const columns: TableColumnsType<AnnouncementVO> = useMemo(
    () => [
      {
        title: "公告标题",
        dataIndex: "title",
        key: "title",
        minwidth: 200,
        ellipsis: true,
      },
      {
        title: "类型",
        dataIndex: "type",
        key: "type",
        width: 110,
        align: "center",
        render: (type: string) => (
          <span className={`type-tag tag-${type}`}>
            {TYPE_LABEL[type] ?? type}
          </span>
        ),
      },
      {
        title: "重要级别",
        dataIndex: "importance",
        key: "importance",
        width: 100,
        align: "center",
        render: (importance: number) =>
          importance === 2 ? (
            <Tag color="danger" bordered={false}>
              重要
            </Tag>
          ) : (
            <Tag bordered={false}>普通</Tag>
          ),
      },
      {
        title: "发送范围",
        dataIndex: "targetScopeLabel",
        key: "targetScopeLabel",
        width: 120,
        align: "center",
        render: (label?: string) => label ?? "-",
      },
      {
        title: "状态",
        dataIndex: "status",
        key: "status",
        width: 100,
        align: "center",
        render: (status: number) => {
          const cfg = STATUS_TAG[status];
          return cfg ? (
            <Tag color={cfg.color} bordered={false}>
              {cfg.label}
            </Tag>
          ) : (
            "-"
          );
        },
      },
      {
        title: "发送时间",
        dataIndex: "sendTime",
        key: "sendTime",
        width: 170,
        align: "center",
        render: (t?: string) => t ?? "-",
      },
      {
        title: "送达数",
        dataIndex: "sentCount",
        key: "sentCount",
        width: 90,
        align: "center",
        render: (c?: number) => c ?? "-",
      },
      {
        title: "创建时间",
        dataIndex: "createTime",
        key: "createTime",
        width: 170,
        align: "center",
      },
      {
        title: "操作",
        key: "action",
        width: 260,
        align: "center",
        fixed: "right",
        render: (_: unknown, record: AnnouncementVO) => (
          <Space size="small">
            {(record.status === 1 || record.status === 2) &&
              hasPerm("notify:announcement:edit") && (
                <Button
                  type="link"
                  size="small"
                  icon={<EditOutlined />}
                  onClick={() => handleEdit(record)}
                >
                  编辑
                </Button>
              )}
            {(record.status === 1 || record.status === 2) &&
              hasPerm("notify:announcement:send") && (
                <Button
                  type="link"
                  size="small"
                  onClick={() => handleSend(record)}
                >
                  发送
                </Button>
              )}
            {record.status === 2 && hasPerm("notify:announcement:cancel") && (
              <Button
                type="link"
                size="small"
                onClick={() => handleCancel(record)}
              >
                取消
              </Button>
            )}
            {hasPerm("notify:announcement:delete") && (
              <Popconfirm
                title={`确认删除公告「${record.title}」吗？删除后不可恢复。`}
                onConfirm={() => handleDelete(record)}
                okText="确定"
                cancelText="取消"
                okType="danger"
              >
                <Button
                  type="link"
                  size="small"
                  danger
                  icon={<DeleteOutlined />}
                >
                  删除
                </Button>
              </Popconfirm>
            )}
          </Space>
        ),
      },
    ],
    [hasPerm, handleEdit, handleSend, handleCancel, handleDelete]
  );

  return (
    <div className="announcement-management-container">
      <Card className="search-card" size="small">
        <Form form={searchForm} layout="inline" onFinish={handleSearch}>
          <Form.Item name="title" label="公告标题">
            <Input placeholder="公告标题" allowClear style={{ width: 200 }} />
          </Form.Item>
          <Form.Item name="type" label="公告类型">
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
              {hasPerm("notify:announcement:add") && (
                <Button
                  type="primary"
                  icon={<PlusOutlined />}
                  onClick={handleAdd}
                >
                  新增公告
                </Button>
              )}
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

      <AnnouncementFormDialog ref={formDialogRef} onSuccess={refreshList} />
    </div>
  );
};

export default AnnouncementManagement;
