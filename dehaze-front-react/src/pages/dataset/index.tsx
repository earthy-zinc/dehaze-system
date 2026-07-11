import { DatasetAPI, type Dataset, type DatasetQuery } from "dehaze-sdk-js";
import DatasetFormDialog, {
  type DatasetFormDialogRef,
} from "@/pages/dataset/components/DatasetFormDialog";
import {
  DeleteOutlined,
  EditOutlined,
  EyeOutlined,
  PlusOutlined,
  ReloadOutlined,
  SearchOutlined,
} from "@ant-design/icons";
import {
  Button,
  Card,
  Form,
  Input,
  message,
  Popconfirm,
  Space,
  Table,
  Tag,
  type TableColumnsType,
} from "antd";
import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useNavigate } from "react-router-dom";

const STATUS_MAP: Record<number, { label: string; color: string }> = {
  1: { label: "启用", color: "green" },
  0: { label: "禁用", color: "default" },
};

/** 清理空 children 数组 */
function cleanDatasets(datasets: Dataset[]): void {
  for (const d of datasets) {
    if (d.children?.length) {
      cleanDatasets(d.children);
    } else if (d.children && Object.keys(d.children).length === 0) {
      delete d.children;
    }
  }
}

export default function DatasetList() {
  const [datasetList, setDatasetList] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(false);
  const [queryParams, setQueryParams] = useState<DatasetQuery>({});
  const [searchForm] = Form.useForm();
  const dialogRef = useRef<DatasetFormDialogRef>(null);
  const [refreshFlag, setRefreshFlag] = useState(0);
  const navigate = useNavigate();

  // ==================== 数据加载 ====================

  const loadData = useCallback(async (params: DatasetQuery) => {
    setLoading(true);
    try {
      const result = await DatasetAPI.getList(params);
      const list = (result as any)?.list || result;
      if (Array.isArray(list)) {
        cleanDatasets(list as Dataset[]);
        setDatasetList(list as Dataset[]);
      } else {
        setDatasetList([]);
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData(queryParams);
  }, [queryParams, refreshFlag]); // eslint-disable-line

  const refreshList = useCallback(() => setRefreshFlag((prev) => prev + 1), []);

  // ==================== 事件处理 ====================

  const handleSearch = useCallback((values: { keywords?: string }) => {
    setQueryParams({ keyword: values.keywords || undefined });
  }, []);

  const handleReset = useCallback(() => {
    searchForm.resetFields();
    setQueryParams({});
  }, [searchForm]);

  const handleAdd = useCallback(() => dialogRef.current?.open("add"), []);

  const handleView = useCallback(
    (id: number) => navigate(`/dataset/${id}`),
    [navigate]
  );

  const handleAddSub = useCallback(
    (record: Dataset) => dialogRef.current?.open("addSub", record),
    []
  );

  const handleEdit = useCallback(
    (record: Dataset) => dialogRef.current?.open("edit", record),
    []
  );

  const handleDelete = useCallback(
    (record: Dataset) => {
      DatasetAPI.deleteById(record.id)
        .then(() => {
          message.success(`数据集「${record.name}」删除成功`);
          refreshList();
        })
        .catch((error) => message.error(error?.message || "删除失败"));
    },
    [refreshList]
  );

  // ==================== 表格列 ====================

  const columns: TableColumnsType<Dataset> = useMemo(
    () => [
      {
        title: "数据集名称",
        dataIndex: "name",
        key: "name",
        width: 200,
        align: "left" as const,
      },
      {
        title: "类型",
        dataIndex: "type",
        key: "type",
        width: 100,
        align: "center",
        render: (text: string) => <Tag>{text}</Tag>,
      },
      {
        title: "状态",
        dataIndex: "status",
        key: "status",
        width: 80,
        align: "center",
        render: (status: number) => {
          const info = STATUS_MAP[status] || {
            label: "未知",
            color: "default",
          };
          return <Tag color={info.color}>{info.label}</Tag>;
        },
      },
      {
        title: "图片数量",
        dataIndex: "total",
        key: "total",
        width: 100,
        align: "center",
        render: (t: number) => t ?? (t === 0 ? 0 : "-"),
      },
      {
        title: "描述",
        dataIndex: "description",
        key: "description",
        render: (text: string) => text || "-",
      },
      {
        title: "操作",
        key: "action",
        width: 280,
        align: "center",
        fixed: "right",
        render: (_: unknown, record: Dataset) => (
          <Space size="small">
            <Button
              type="link"
              size="small"
              icon={<EyeOutlined />}
              onClick={() => handleView(record.id)}
            >
              查看
            </Button>
            <Button
              type="link"
              size="small"
              icon={<PlusOutlined />}
              onClick={() => handleAddSub(record)}
            >
              新增下级
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
              title={`确认删除数据集「${record.name}」吗？`}
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
    [handleView, handleAddSub, handleEdit, handleDelete]
  );

  // ==================== 渲染 ====================

  return (
    <div className="app-container">
      <Card size="small" style={{ marginBottom: 12 }}>
        <Form form={searchForm} layout="inline" onFinish={handleSearch}>
          <Form.Item name="keywords" label="数据集名称">
            <Input placeholder="数据集名称" allowClear style={{ width: 200 }} />
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
            </Space>
          </Form.Item>
        </Form>
      </Card>

      <Card size="small" style={{ overflowX: "hidden" }}>
        <Table
          columns={columns}
          dataSource={datasetList}
          rowKey={(record) => record.id}
          loading={loading}
          expandable={{ defaultExpandAllRows: true, indentSize: 30 }}
          pagination={false}
          scroll={{ x: 1000 }}
        />
      </Card>

      <DatasetFormDialog ref={dialogRef} onSuccess={refreshList} />
    </div>
  );
}
