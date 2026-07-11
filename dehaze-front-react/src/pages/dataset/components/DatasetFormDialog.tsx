import {
  DatasetAPI,
  type Dataset,
  type DatasetAddForm,
  type DatasetUpdateForm,
} from "dehaze-sdk-js";
import { Form, Input, Modal, Radio, TreeSelect, message } from "antd";
import React, {
  forwardRef,
  useCallback,
  useImperativeHandle,
  useState,
} from "react";

/** 递归转换数据集树为 TreeSelect 格式 */
function buildTreeSelect(datasets: Dataset[]): any[] {
  return datasets.map((d) => ({
    title: d.name,
    value: d.id,
    children: d.children?.length ? buildTreeSelect(d.children) : undefined,
  }));
}

export interface DatasetFormDialogRef {
  open: (type: "add" | "edit" | "addSub", record?: Dataset) => void;
}

interface Props {
  onSuccess?: () => void;
}

const DatasetFormDialog = forwardRef<DatasetFormDialogRef, Props>(
  ({ onSuccess }, ref) => {
    const [visible, setVisible] = useState(false);
    const [dialogType, setDialogType] = useState<"add" | "edit" | "addSub">(
      "add"
    );
    const [confirmLoading, setConfirmLoading] = useState(false);
    const [form] = Form.useForm();
    const [treeData, setTreeData] = useState<any[]>([]);

    const loadTree = useCallback(async () => {
      try {
        const data = await DatasetAPI.getList();
        setTreeData([
          { title: "顶级数据集", value: 0 },
          ...buildTreeSelect((data as any)?.list || data || []),
        ]);
      } catch {
        setTreeData([{ title: "顶级数据集", value: 0 }]);
      }
    }, []);

    const open = useCallback(
      async (type: "add" | "edit" | "addSub", record?: Dataset) => {
        setDialogType(type);
        setVisible(true);
        loadTree();

        if (type === "add") {
          form.resetFields();
          form.setFieldsValue({ parentId: 0, status: 1 });
        } else if (type === "addSub" && record) {
          form.resetFields();
          form.setFieldsValue({ parentId: record.id, status: 1 });
        } else if (type === "edit" && record) {
          form.resetFields();
          form.setFieldsValue({
            id: record.id,
            parentId: record.parentId ?? 0,
            name: record.name,
            type: record.type,
            description: record.description,
            path: record.path,
            status: record.status ?? 1,
          });
        }
      },
      [form, loadTree]
    );

    useImperativeHandle(ref, () => ({ open }), [open]);

    const handleCancel = useCallback(() => {
      setVisible(false);
      form.resetFields();
    }, [form]);

    const handleSubmit = useCallback(async () => {
      try {
        const values = await form.validateFields();
        setConfirmLoading(true);

        if (dialogType === "edit") {
          const id = form.getFieldValue("id");
          const updateData: DatasetUpdateForm = {
            name: values.name,
            type: values.type,
            description: values.description,
            path: values.path,
            status: String(values.status),
          };
          await DatasetAPI.update(id, updateData);
          message.success("修改数据集成功");
        } else {
          const addData: DatasetAddForm = { parentId: values.parentId ?? 0 };
          if (values.name) addData.name = values.name;
          if (values.type) addData.type = values.type;
          if (values.description) addData.description = values.description;
          if (values.path) addData.path = values.path;
          addData.status = String(values.status ?? 1);
          await DatasetAPI.add(addData);
          message.success("新增数据集成功");
        }

        handleCancel();
        onSuccess?.();
      } catch (error: any) {
        if (error?.errorFields) return;
        message.error(error?.message || "操作失败");
      } finally {
        setConfirmLoading(false);
      }
    }, [form, dialogType, handleCancel, onSuccess]);

    const title = dialogType === "edit" ? "修改数据集" : "新增数据集";

    return (
      <Modal
        title={title}
        open={visible}
        width={600}
        confirmLoading={confirmLoading}
        okText="保存"
        cancelText="取消"
        destroyOnClose
        onOk={handleSubmit}
        onCancel={handleCancel}
      >
        <Form
          form={form}
          layout="vertical"
          colon={false}
          validateTrigger="onBlur"
        >
          <Form.Item name="parentId" label="上级数据集">
            <TreeSelect
              treeData={treeData}
              placeholder="请选择上级数据集（为空则顶级）"
              treeDefaultExpandAll
              allowClear
            />
          </Form.Item>
          <Form.Item
            name="name"
            label="数据集名称"
            rules={[{ required: true, message: "请输入数据集名称" }]}
          >
            <Input placeholder="请输入数据集名称" />
          </Form.Item>
          <Form.Item
            name="type"
            label="数据集类型"
            rules={[{ required: true, message: "请输入数据集类型" }]}
          >
            <Input
              placeholder="请输入数据集类型（如 training/test）"
              disabled={dialogType === "edit"}
            />
          </Form.Item>
          <Form.Item
            name="path"
            label="存储路径"
            rules={[{ required: true, message: "请输入数据集存储路径" }]}
          >
            <Input placeholder="请输入数据集存储路径（如 /data/training）" />
          </Form.Item>
          <Form.Item
            name="description"
            label="数据集描述"
            rules={[{ max: 500, message: "描述不能超过500字" }]}
          >
            <Input.TextArea placeholder="请输入数据集描述" rows={3} />
          </Form.Item>
          <Form.Item name="status" label="状态">
            <Radio.Group>
              <Radio value={1}>启用</Radio>
              <Radio value={0}>禁用</Radio>
            </Radio.Group>
          </Form.Item>
        </Form>
      </Modal>
    );
  }
);

DatasetFormDialog.displayName = "DatasetFormDialog";
export default DatasetFormDialog;
