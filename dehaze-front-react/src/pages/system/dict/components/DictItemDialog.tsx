import { DictAPI, type DictForm, type DictPageVO, type DictQuery } from "dehaze-sdk-js";
import {
  Button, Form, Input, InputNumber, message, Modal, Popconfirm,
  Radio, Space, Table, Tag, type TableColumnsType,
} from "antd";
import { DeleteOutlined, EditOutlined, PlusOutlined } from "@ant-design/icons";
import React, { forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useState } from "react";

export interface DictItemDialogRef {
  open: (typeCode: string, typeName: string) => void;
}

const STATUS_MAP: Record<number, { label: string; color: string }> = {
  1: { label: "启用", color: "green" },
  0: { label: "禁用", color: "default" },
};

const DictItemDialog = forwardRef<DictItemDialogRef>((_props, ref) => {
  const [visible, setVisible] = useState(false);
  const [typeCode, setTypeCode] = useState("");
  const [typeName, setTypeName] = useState("");
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<DictPageVO[]>([]);
  const [total, setTotal] = useState(0);
  const [query, setQuery] = useState<DictQuery>({ pageNum: 1, pageSize: 10 });
  const [selectedRowKeys, setSelectedRowKeys] = useState<React.Key[]>([]);

  // form state
  const [formVisible, setFormVisible] = useState(false);
  const [formType, setFormType] = useState<"add" | "edit">("add");
  const [confirmLoading, setConfirmLoading] = useState(false);
  const [dictForm] = Form.useForm<DictForm>();

  const loadData = useCallback(async () => {
    if (!typeCode) return;
    setLoading(true);
    try {
      const result = await DictAPI.getDictPage({ ...query, typeCode });
      setData(result.list || []);
      setTotal(result.total || 0);
    } finally { setLoading(false); }
  }, [query, typeCode]);

  useEffect(() => { loadData(); }, [loadData]);

  const open = useCallback((code: string, name: string) => {
    setTypeCode(code);
    setTypeName(name);
    setQuery({ pageNum: 1, pageSize: 10 });
    setSelectedRowKeys([]);
    setVisible(true);
  }, []);

  useImperativeHandle(ref, () => ({ open }), [open]);

  // item form
  const openForm = useCallback((t: "add" | "edit", record?: DictPageVO) => {
    setFormType(t);
    setFormVisible(true);
    if (t === "add") {
      dictForm.resetFields();
      dictForm.setFieldsValue({ typeCode, status: 1, sort: 1, defaulted: 0 });
    } else if (record?.id) {
      dictForm.resetFields();
      DictAPI.getDictFormData(record.id).then((d) => {
        dictForm.setFieldsValue({ ...d, typeCode });
      }).catch(() => message.error("获取字典数据失败"));
    }
  }, [dictForm, typeCode]);

  const handleFormSubmit = useCallback(async () => {
    try {
      const values = await dictForm.validateFields();
      setConfirmLoading(true);
      if (formType === "add") {
        await DictAPI.addDict(values);
        message.success("新增字典数据成功");
      } else {
        const id = dictForm.getFieldValue("id");
        await DictAPI.updateDict(id, values);
        message.success("修改字典数据成功");
      }
      setFormVisible(false);
      dictForm.resetFields();
      loadData();
    } catch (error: any) {
      if (error?.errorFields) return;
      message.error(error?.message || "操作失败");
    } finally { setConfirmLoading(false); }
  }, [dictForm, formType, loadData]);

  const handleDelete = useCallback((record: DictPageVO) => {
    DictAPI.deleteDictByIds(String(record.id)).then(() => {
      message.success("删除成功");
      setSelectedRowKeys([]);
      loadData();
    }).catch((error) => message.error(error?.message || "删除失败"));
  }, [loadData]);

  // 批量删除字典数据
  const handleBatchDelete = useCallback(() => {
    Modal.confirm({
      title: "批量删除",
      content: `确认删除选中的 ${selectedRowKeys.length} 个字典数据吗？删除后不可恢复。`,
      okText: "确定", cancelText: "取消", okType: "danger",
      onOk: () => DictAPI.deleteDictByIds(selectedRowKeys.join(",")).then(() => {
        message.success(`成功删除 ${selectedRowKeys.length} 个字典数据`);
        setSelectedRowKeys([]);
        loadData();
      }).catch((error) => { message.error(error?.message || "删除失败"); return Promise.reject(error); }),
    });
  }, [selectedRowKeys, loadData]);

  const columns: TableColumnsType<DictPageVO> = useMemo(() => [
    { title: "数据名称", dataIndex: "name", key: "name", width: 150 },
    { title: "数据值", dataIndex: "value", key: "value", width: 120, align: "center" as const },
    {
      title: "状态", dataIndex: "status", key: "status", width: 80, align: "center",
      render: (s: number) => <Tag color={STATUS_MAP[s]?.color}>{STATUS_MAP[s]?.label}</Tag>,
    },
    { title: "排序", dataIndex: "sort", key: "sort", width: 80, align: "center" as const },
    { title: "备注", dataIndex: "remark", key: "remark", width: 150, render: (t: string) => t || "-" },
    { title: "创建时间", dataIndex: "createTime" as any, key: "createTime", width: 180, align: "center" },
    {
      title: "操作", key: "action", width: 140, align: "center", fixed: "right" as const,
      render: (_: unknown, record: DictPageVO) => (
        <Space size="small">
          <Button type="link" size="small" icon={<EditOutlined />} onClick={() => openForm("edit", record)}>编辑</Button>
          <Popconfirm title={`确认删除字典数据「${record.name}」吗？删除后不可恢复。`} onConfirm={() => handleDelete(record)} okText="确定" cancelText="取消" okType="danger">
            <Button type="link" size="small" danger icon={<DeleteOutlined />}>删除</Button>
          </Popconfirm>
        </Space>
      ),
    },
  ], [openForm, handleDelete]);

  // 行选择配置
  const rowSelection = useMemo(() => ({
    selectedRowKeys,
    onChange: (keys: React.Key[]) => setSelectedRowKeys(keys),
  }), [selectedRowKeys]);

  return (
    <>
      <Modal
        title={`字典数据管理 - ${typeName}`}
        open={visible} width={900} footer={null}
        onCancel={() => setVisible(false)} destroyOnClose
      >
        <div style={{ marginBottom: 12 }}>
          <Space>
            <Button type="primary" icon={<PlusOutlined />} onClick={() => openForm("add")}>新增</Button>
            <Button danger icon={<DeleteOutlined />} disabled={selectedRowKeys.length === 0} onClick={handleBatchDelete}>删除</Button>
          </Space>
        </div>
        <Table
          columns={columns} dataSource={data} rowKey={(r) => r.id ?? Math.random()}
          loading={loading} size="small" rowSelection={rowSelection}
          pagination={{
            current: query.pageNum, pageSize: query.pageSize, total,
            showSizeChanger: true, showTotal: (t) => `共 ${t} 条`,
            onChange: (p, ps) => setQuery((prev) => ({ ...prev, pageNum: p, pageSize: ps })),
          }}
        />
      </Modal>

      <Modal
        title={formType === "add" ? "新增字典数据" : "修改字典数据"}
        open={formVisible} width={500} confirmLoading={confirmLoading}
        okText="保存" cancelText="取消" destroyOnClose onOk={handleFormSubmit}
        onCancel={() => { setFormVisible(false); dictForm.resetFields(); }}
      >
        <Form form={dictForm} layout="vertical" colon={false} validateTrigger="onBlur">
          <Form.Item name="name" label="数据名称" rules={[{ required: true, message: "请输入数据名称" }]}>
            <Input placeholder="请输入数据名称" />
          </Form.Item>
          <Form.Item name="value" label="数据值" rules={[{ required: true, message: "请输入数据值" }]}>
            <Input placeholder="请输入数据值" />
          </Form.Item>
          <Form.Item name="sort" label="排序" rules={[{ required: true, message: "请输入排序值" }]}>
            <InputNumber min={1} style={{ width: "100%" }} />
          </Form.Item>
          <Form.Item name="remark" label="备注">
            <Input.TextArea placeholder="请输入备注" rows={2} />
          </Form.Item>
          <Form.Item name="defaulted" label="是否默认">
            <Radio.Group>
              <Radio value={1}>是</Radio>
              <Radio value={0}>否</Radio>
            </Radio.Group>
          </Form.Item>
          <Form.Item name="status" label="状态">
            <Radio.Group>
              <Radio value={1}>启用</Radio>
              <Radio value={0}>禁用</Radio>
            </Radio.Group>
          </Form.Item>
        </Form>
      </Modal>
    </>
  );
});

DictItemDialog.displayName = "DictItemDialog";
export default DictItemDialog;
