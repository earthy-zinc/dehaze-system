import { AlgorithmAPI, type Algorithm, type AlgorithmQuery } from "dehaze-sdk-js";
import AlgorithmFormDialog, { type AlgorithmFormDialogRef } from "@/pages/algorithm/components/AlgorithmFormDialog";
import {
  DeleteOutlined, EditOutlined, PlusOutlined, ReloadOutlined, SearchOutlined,
} from "@ant-design/icons";
import {
  Button, Card, Form, Input, message, Popconfirm, Space, Table, Tag,
  type TableColumnsType,
} from "antd";
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";

/** 算法状态映射 */
const STATUS_MAP: Record<number, { label: string; color: string }> = {
  0: { label: "草稿", color: "default" },
  1: { label: "测试中", color: "processing" },
  2: { label: "待审核", color: "warning" },
  3: { label: "已发布", color: "green" },
  4: { label: "已停用", color: "error" },
  5: { label: "已归档", color: "default" },
};

/** 清理空 children */
function cleanAlgorithms(algorithms: Algorithm[]): void {
  for (const a of algorithms) {
    if (a.children?.length) { cleanAlgorithms(a.children); }
    else if (a.children && Object.keys(a.children).length === 0) { delete a.children; }
  }
}

export default function AlgorithmList(): React.JSX.Element {
  const [algorithmList, setAlgorithmList] = useState<Algorithm[]>([]);
  const [loading, setLoading] = useState(false);
  const [queryParams, setQueryParams] = useState<AlgorithmQuery>({});
  const [searchForm] = Form.useForm();
  const dialogRef = useRef<AlgorithmFormDialogRef>(null);
  const [refreshFlag, setRefreshFlag] = useState(0);

  // ==================== 数据加载 ====================

  const loadData = useCallback(async (params: AlgorithmQuery) => {
    setLoading(true);
    try {
      const data = await AlgorithmAPI.getList(params);
      if (Array.isArray(data)) {
        cleanAlgorithms(data);
        setAlgorithmList(data);
      } else { setAlgorithmList([]); }
    } finally { setLoading(false); }
  }, []);

  useEffect(() => { loadData(queryParams); }, [queryParams, refreshFlag]); // eslint-disable-line

  const refreshList = useCallback(() => setRefreshFlag((prev) => prev + 1), []);

  // ==================== 事件处理 ====================

  const handleSearch = useCallback((values: { keywords?: string }) => {
    setQueryParams({ keywords: values.keywords || undefined });
  }, []);

  const handleReset = useCallback(() => { searchForm.resetFields(); setQueryParams({}); }, [searchForm]);

  const handleAdd = useCallback(() => dialogRef.current?.open("add"), []);
  const handleAddSub = useCallback((record: Algorithm) => dialogRef.current?.open("addSub", record), []);
  const handleEdit = useCallback((record: Algorithm) => dialogRef.current?.open("edit", record), []);

  const handleDelete = useCallback(
    (record: Algorithm) => {
      AlgorithmAPI.deleteByIds([String(record.id)]).then(() => {
        message.success(`算法「${record.name}」删除成功`);
        refreshList();
      }).catch((error) => message.error(error?.message || "删除失败"));
    },
    [refreshList]
  );

  // ==================== 表格列 ====================

  const columns: TableColumnsType<Algorithm> = useMemo(() => [
    {
      title: "算法名称", dataIndex: "name", key: "name", width: 200, align: "left" as const,
    },
    {
      title: "类型", dataIndex: "type", key: "type", width: 120, align: "center",
      render: (text: string) => <Tag>{text}</Tag>,
    },
    {
      title: "状态", dataIndex: "status", key: "status", width: 90, align: "center",
      render: (status: number) => {
        const info = STATUS_MAP[status] || { label: "未知", color: "default" };
        return <Tag color={info.color}>{info.label}</Tag>;
      },
    },
    {
      title: "代码导入路径", dataIndex: "importPath", key: "importPath", width: 160,
      render: (text: string) => text || "-",
    },
    {
      title: "FLOPs", dataIndex: "flops", key: "flops", width: 100, align: "center",
      render: (text: string) => text || "-",
    },
    {
      title: "参数量", dataIndex: "params", key: "params", width: 100, align: "center",
      render: (text: string) => text || "-",
    },
    {
      title: "描述", dataIndex: "description", key: "description",
      render: (text: string) => text || "-",
    },
    {
      title: "操作", key: "action", width: 220, align: "center", fixed: "right",
      render: (_: unknown, record: Algorithm) => (
        <Space size="small">
          <Button type="link" size="small" icon={<PlusOutlined />} onClick={() => handleAddSub(record)}>新增下级</Button>
          <Button type="link" size="small" icon={<EditOutlined />} onClick={() => handleEdit(record)}>编辑</Button>
          <Popconfirm
            title={`确认删除算法「${record.name}」吗？`}
            onConfirm={() => handleDelete(record)} okText="确定" cancelText="取消" okType="danger"
          >
            <Button type="link" size="small" danger icon={<DeleteOutlined />}>删除</Button>
          </Popconfirm>
        </Space>
      ),
    },
  ], [handleAddSub, handleEdit, handleDelete]);

  // ==================== 渲染 ====================

  return (
    <div className="app-container">
      <Card size="small" style={{ marginBottom: 12 }}>
        <Form form={searchForm} layout="inline" onFinish={handleSearch}>
          <Form.Item name="keywords" label="关键字">
            <Input placeholder="算法名称" allowClear style={{ width: 200 }} />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" icon={<SearchOutlined />}>搜索</Button>
              <Button htmlType="reset" icon={<ReloadOutlined />} onClick={handleReset}>重置</Button>
              <Button type="primary" icon={<PlusOutlined />} onClick={handleAdd}>新增</Button>
            </Space>
          </Form.Item>
        </Form>
      </Card>

      <Card size="small" style={{ overflowX: "hidden" }}>
        <Table
          columns={columns} dataSource={algorithmList}
          rowKey={(record) => record.id} loading={loading}
          expandable={{ defaultExpandAllRows: true, indentSize: 30 }}
          pagination={false} scroll={{ x: 1200 }}
        />
      </Card>

      <AlgorithmFormDialog ref={dialogRef} onSuccess={refreshList} />
    </div>
  );
}
