import {
  DictAPI,
  type DictTypePageVO,
  type DictTypeQuery,
} from "dehaze-sdk-js";
import {
  Button,
  Card,
  Form,
  Input,
  message,
  Modal,
  Popconfirm,
  Space,
  Table,
  Tag,
  type TableColumnsType,
} from "antd";
import {
  DeleteOutlined,
  EditOutlined,
  PlusOutlined,
  ReloadOutlined,
  SearchOutlined,
  UnorderedListOutlined,
} from "@ant-design/icons";
import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import DictItemDialog, {
  type DictItemDialogRef,
} from "./components/DictItemDialog";
import DictTypeFormDialog, {
  type DictTypeFormDialogRef,
} from "./components/DictTypeFormDialog";
import "./index.scss";

const STATUS_MAP: Record<number, { label: string; color: string }> = {
  1: { label: "启用", color: "green" },
  0: { label: "禁用", color: "default" },
};

const DictManagement: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<DictTypePageVO[]>([]);
  const [total, setTotal] = useState(0);
  const [searchForm] = Form.useForm();
  const [query, setQuery] = useState<DictTypeQuery>({
    pageNum: 1,
    pageSize: 10,
  });

  const typeDialogRef = useRef<DictTypeFormDialogRef>(null);
  const itemDialogRef = useRef<DictItemDialogRef>(null);
  const [refreshFlag, setRefreshFlag] = useState(0);
  const [selectedRowKeys, setSelectedRowKeys] = useState<React.Key[]>([]);

  const loadData = useCallback(async (params: DictTypeQuery) => {
    setLoading(true);
    try {
      const result = await DictAPI.getDictTypePage(params);
      setData(result.list || []);
      setTotal(result.total || 0);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadData(query); }, [query, refreshFlag]); // eslint-disable-line

  const refreshList = useCallback(() => {
    setRefreshFlag((prev) => prev + 1);
    setSelectedRowKeys([]);
  }, []);

  const handleSearch = useCallback((values: { keywords?: string }) => {
    setSelectedRowKeys([]);
    setQuery((prev) => ({
      ...prev,
      pageNum: 1,
      keywords: values.keywords || undefined,
    }));
  }, []);

  const handleReset = useCallback(() => {
    searchForm.resetFields();
    setSelectedRowKeys([]);
    setQuery({ pageNum: 1, pageSize: 10 });
  }, [searchForm]);
  const handleAdd = useCallback(() => typeDialogRef.current?.open("add"), []);
  const handleEdit = useCallback(
    (record: DictTypePageVO) => typeDialogRef.current?.open("edit", record),
    []
  );
  const handleDelete = useCallback(
    (record: DictTypePageVO) => {
      DictAPI.deleteDictTypes(String(record.id))
        .then(() => {
          message.success("删除成功");
          refreshList();
        })
        .catch((e) => message.error(e?.message || "删除失败"));
    },
    [refreshList]
  );
  // 批量删除字典类型
  const handleBatchDelete = useCallback(() => {
    Modal.confirm({
      title: "批量删除",
      content: `确认删除选中的 ${selectedRowKeys.length} 个字典类型吗？删除后不可恢复。`,
      okText: "确定",
      cancelText: "取消",
      okType: "danger",
      onOk: () =>
        DictAPI.deleteDictTypes(selectedRowKeys.join(","))
          .then(() => {
            message.success(`成功删除 ${selectedRowKeys.length} 个字典类型`);
            refreshList();
          })
          .catch((e) => {
            message.error(e?.message || "删除失败");
            return Promise.reject(e);
          }),
    });
  }, [selectedRowKeys, refreshList]);
  const handleManageItems = useCallback(
    (record: DictTypePageVO) =>
      itemDialogRef.current?.open(record.code, record.name),
    []
  );

  const columns: TableColumnsType<DictTypePageVO> = useMemo(
    () => [
      { title: "类型名称", dataIndex: "name", key: "name", width: 200 },
      { title: "类型编码", dataIndex: "code", key: "code", width: 150 },
      {
        title: "状态",
        dataIndex: "status",
        key: "status",
        width: 80,
        align: "center",
        render: (s: number) => (
          <Tag color={STATUS_MAP[s]?.color}>{STATUS_MAP[s]?.label}</Tag>
        ),
      },
      {
        title: "备注",
        dataIndex: "remark",
        key: "remark",
        width: 200,
        render: (t: string) => t || "-",
      },
      {
        title: "创建时间",
        dataIndex: "createTime" as any,
        key: "createTime",
        width: 180,
        align: "center",
      },
      {
        title: "操作",
        key: "action",
        width: 220,
        align: "center",
        fixed: "right",
        render: (_: unknown, record: DictTypePageVO) => (
          <Space size="small">
            <Button
              type="link"
              size="small"
              icon={<UnorderedListOutlined />}
              onClick={() => handleManageItems(record)}
            >
              管理
            </Button>
            <Button
              type="link"
              size="small"
              icon={<EditOutlined />}
              onClick={() => handleEdit(record)}
            >
              编辑
            </Button>
            <Popconfirm
              title={`确认删除字典类型「${record.name}」吗？删除后不可恢复。`}
              onConfirm={() => handleDelete(record)}
              okText="确定"
              cancelText="取消"
              okType="danger"
            >
              <Button type="link" size="small" danger icon={<DeleteOutlined />}>
                删除
              </Button>
            </Popconfirm>
          </Space>
        ),
      },
    ],
    [handleManageItems, handleEdit, handleDelete]
  );

  // 行选择配置
  const rowSelection = useMemo(
    () => ({
      selectedRowKeys,
      onChange: (keys: React.Key[]) => setSelectedRowKeys(keys),
    }),
    [selectedRowKeys]
  );

  return (
    <div className="dict-management-container">
      <Card className="search-card" size="small">
        <Form form={searchForm} layout="inline" onFinish={handleSearch}>
          <Form.Item name="keywords" label="关键字">
            <Input
              placeholder="类型名称/编码"
              allowClear
              style={{ width: 200 }}
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
              <Button
                type="primary"
                icon={<PlusOutlined />}
                onClick={handleAdd}
              >
                新增
              </Button>
              <Button
                danger
                icon={<DeleteOutlined />}
                disabled={selectedRowKeys.length === 0}
                onClick={handleBatchDelete}
              >
                删除
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Card>
      <Card className="table-card" size="small">
        <Table
          columns={columns}
          dataSource={data}
          rowKey={(r) => r.id}
          loading={loading}
          scroll={{ x: 1000 }}
          rowSelection={rowSelection}
          pagination={{
            current: query.pageNum,
            pageSize: query.pageSize,
            total,
            showSizeChanger: true,
            showQuickJumper: true,
            pageSizeOptions: ["10", "20", "50", "100"],
            showTotal: (t) => `共 ${t} 条`,
            onChange: (p, ps) =>
              setQuery((prev) => ({ ...prev, pageNum: p, pageSize: ps })),
          }}
        />
      </Card>
      <DictTypeFormDialog ref={typeDialogRef} onSuccess={refreshList} />
      <DictItemDialog ref={itemDialogRef} />
    </div>
  );
};

export default DictManagement;
