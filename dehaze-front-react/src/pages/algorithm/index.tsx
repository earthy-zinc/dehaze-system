import {
  AlgorithmAPI,
  type Algorithm,
  type AlgorithmQuery,
} from "dehaze-sdk-js";
import AlgorithmFormDialog, {
  type AlgorithmFormDialogRef,
} from "@/pages/algorithm/components/AlgorithmFormDialog";
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
  Descriptions,
  Form,
  Input,
  message,
  Modal,
  Popconfirm,
  Space,
  Switch,
  Table,
  Tag,
  type TableColumnsType,
} from "antd";
import { useDebounceFn } from "ahooks";
import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

/** 清理空 children */
function cleanAlgorithms(algorithms: Algorithm[]): void {
  for (const a of algorithms) {
    if (a.children?.length) {
      cleanAlgorithms(a.children);
    } else if (a.children && Object.keys(a.children).length === 0) {
      delete a.children;
    }
  }
}

export default function AlgorithmList(): React.JSX.Element {
  const [algorithmList, setAlgorithmList] = useState<Algorithm[]>([]);
  const [loading, setLoading] = useState(false);
  const [queryParams, setQueryParams] = useState<AlgorithmQuery>({});
  const [searchForm] = Form.useForm();
  const dialogRef = useRef<AlgorithmFormDialogRef>(null);
  const [refreshFlag, setRefreshFlag] = useState(0);
  const [detailVisible, setDetailVisible] = useState(false);
  const [detailRecord, setDetailRecord] = useState<Algorithm | null>(null);

  // ==================== 数据加载 ====================

  const loadData = useCallback(async (params: AlgorithmQuery) => {
    setLoading(true);
    try {
      const data = await AlgorithmAPI.getList(params);
      if (Array.isArray(data)) {
        cleanAlgorithms(data);
        setAlgorithmList(data);
      } else {
        setAlgorithmList([]);
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData(queryParams);
  }, [queryParams, refreshFlag]); // eslint-disable-line

  const refreshList = useCallback(() => setRefreshFlag((prev) => prev + 1), []);

  // ==================== 搜索防抖（300ms） ====================

  const { run: debouncedSearch } = useDebounceFn(
    (keywords: string) => {
      setQueryParams({ keywords: keywords || undefined });
    },
    { wait: 300 }
  );

  const handleSearchChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      debouncedSearch(e.target.value);
    },
    [debouncedSearch]
  );

  const handleReset = useCallback(() => {
    searchForm.resetFields();
    setQueryParams({});
  }, [searchForm]);

  /** 点击搜索按钮：读取表单当前值并触发查询 */
  const handleSearch = useCallback(() => {
    const keywords = searchForm.getFieldValue("keywords") as string | undefined;
    setQueryParams({ keywords: keywords || undefined });
  }, [searchForm]);

  // ==================== 事件处理 ====================

  const handleAdd = useCallback(() => dialogRef.current?.open("add"), []);
  const handleAddSub = useCallback(
    (record: Algorithm) => dialogRef.current?.open("addSub", record),
    []
  );
  const handleEdit = useCallback(
    (record: Algorithm) => dialogRef.current?.open("edit", record),
    []
  );

  /** 查看算法详情 */
  const handleViewDetail = useCallback((record: Algorithm) => {
    setDetailRecord(record);
    setDetailVisible(true);
  }, []);

  /** 切换算法启用状态 */
  const handleStatusChange = useCallback(
    async (checked: boolean, record: Algorithm) => {
      const status = checked ? 1 : 0;
      try {
        await AlgorithmAPI.updateStatus(record.id, status);
        message.success(`算法「${record.name}」已${checked ? "启用" : "禁用"}`);
        refreshList();
      } catch (error: any) {
        message.error(error?.message || "状态更新失败");
      }
    },
    [refreshList]
  );

  const handleDelete = useCallback(
    (record: Algorithm) => {
      AlgorithmAPI.deleteByIds([String(record.id)])
        .then(() => {
          message.success(`算法「${record.name}」删除成功`);
          refreshList();
        })
        .catch((error) => message.error(error?.message || "删除失败"));
    },
    [refreshList]
  );

  // ==================== 表格列 ====================

  const columns: TableColumnsType<Algorithm> = useMemo(
    () => [
      {
        title: "算法名称",
        dataIndex: "name",
        key: "name",
        width: 200,
        align: "left" as const,
        render: (text: string, record: Algorithm) => (
          <Button
            type="link"
            size="small"
            style={{ padding: 0 }}
            onClick={() => handleViewDetail(record)}
          >
            {text}
          </Button>
        ),
      },
      {
        title: "类型",
        dataIndex: "type",
        key: "type",
        width: 120,
        align: "center",
        render: (text: string) => <Tag>{text}</Tag>,
      },
      {
        title: "状态",
        dataIndex: "status",
        key: "status",
        width: 100,
        align: "center",
        render: (status: number, record: Algorithm) => (
          <Switch
            checked={status === 1}
            checkedChildren="启用"
            unCheckedChildren="禁用"
            onChange={(checked) => handleStatusChange(checked, record)}
          />
        ),
      },
      {
        title: "代码导入路径",
        dataIndex: "importPath",
        key: "importPath",
        width: 160,
        render: (text: string) => text || "-",
      },
      {
        title: "FLOPs",
        dataIndex: "flops",
        key: "flops",
        width: 100,
        align: "center",
        render: (text: string) => text || "-",
      },
      {
        title: "参数量",
        dataIndex: "params",
        key: "params",
        width: 100,
        align: "center",
        render: (text: string) => text || "-",
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
        width: 220,
        align: "center",
        fixed: "right",
        render: (_: unknown, record: Algorithm) => (
          <Space size="small">
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
              title={`确认删除算法「${record.name}」吗？删除后不可恢复。`}
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
    [
      handleAddSub,
      handleEdit,
      handleDelete,
      handleStatusChange,
      handleViewDetail,
    ]
  );

  // ==================== 渲染 ====================

  return (
    <div className="app-container">
      <Card className="search-card" size="small">
        <Form form={searchForm} layout="inline" onFinish={handleSearch}>
          <Form.Item name="keywords" label="关键字">
            <Input
              placeholder="算法名称"
              allowClear
              style={{ width: 200 }}
              onChange={handleSearchChange}
              onPressEnter={handleSearch}
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
            </Space>
          </Form.Item>
        </Form>
      </Card>

      <Card className="table-card" size="small">
        <Table
          columns={columns}
          dataSource={algorithmList}
          rowKey={(record) => record.id}
          loading={loading}
          expandable={{ defaultExpandAllRows: true, indentSize: 30 }}
          pagination={false}
          scroll={{ x: 1200 }}
        />
      </Card>

      <AlgorithmFormDialog ref={dialogRef} onSuccess={refreshList} />

      {/* 算法详情弹窗 */}
      <Modal
        title="算法详情"
        open={detailVisible}
        width={680}
        footer={null}
        onCancel={() => setDetailVisible(false)}
        destroyOnHidden
      >
        {detailRecord && (
          <Descriptions column={2} bordered size="small">
            <Descriptions.Item label="算法名称">
              {detailRecord.name}
            </Descriptions.Item>
            <Descriptions.Item label="算法类型">
              <Tag>{detailRecord.type}</Tag>
            </Descriptions.Item>
            <Descriptions.Item label="状态">
              <Tag color={detailRecord.status === 1 ? "green" : "default"}>
                {detailRecord.status === 1 ? "启用" : "禁用"}
              </Tag>
            </Descriptions.Item>
            <Descriptions.Item label="代码导入路径">
              {detailRecord.importPath || "-"}
            </Descriptions.Item>
            <Descriptions.Item label="FLOPs">
              {detailRecord.flops || "-"}
            </Descriptions.Item>
            <Descriptions.Item label="参数量">
              {detailRecord.params || "-"}
            </Descriptions.Item>
            <Descriptions.Item label="算法大小">
              {detailRecord.size || "-"}
            </Descriptions.Item>
            <Descriptions.Item label="创建时间">
              {detailRecord.createTime || "-"}
            </Descriptions.Item>
            <Descriptions.Item label="算法描述" span={2}>
              {detailRecord.description || "-"}
            </Descriptions.Item>
          </Descriptions>
        )}
      </Modal>
    </div>
  );
}
