import { DictAPI, type DictTypeForm } from "dehaze-sdk-js";
import { Form, Input, Modal, Radio, message } from "antd";
import React, {
  forwardRef,
  useCallback,
  useImperativeHandle,
  useState,
} from "react";

export interface DictTypeFormDialogRef {
  open: (type: "add" | "edit", record?: { id: number }) => void;
}

interface Props {
  onSuccess?: () => void;
}

const DictTypeFormDialog = forwardRef<DictTypeFormDialogRef, Props>(
  ({ onSuccess }, ref) => {
    const [visible, setVisible] = useState(false);
    const [dialogType, setDialogType] = useState<"add" | "edit">("add");
    const [confirmLoading, setConfirmLoading] = useState(false);
    const [form] = Form.useForm<DictTypeForm>();

    const open = useCallback(
      async (type: "add" | "edit", record?: { id: number }) => {
        setDialogType(type);
        setVisible(true);
        if (type === "add") {
          form.resetFields();
          form.setFieldsValue({ status: 1 });
        } else if (record?.id) {
          form.resetFields();
          try {
            const data = await DictAPI.getDictTypeForm(record.id);
            form.setFieldsValue(data);
          } catch {
            message.error("获取字典类型信息失败");
          }
        }
      },
      [form]
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
        if (dialogType === "add") {
          await DictAPI.addDictType(values);
          message.success("新增字典类型成功");
        } else {
          const id = form.getFieldValue("id");
          await DictAPI.updateDictType(id, values);
          message.success("修改字典类型成功");
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

    return (
      <Modal
        title={dialogType === "add" ? "新增字典类型" : "修改字典类型"}
        open={visible}
        width={500}
        confirmLoading={confirmLoading}
        okText="保存"
        cancelText="取消"
        destroyOnHidden
        onOk={handleSubmit}
        onCancel={handleCancel}
      >
        <Form
          form={form}
          layout="vertical"
          colon={false}
          validateTrigger="onBlur"
        >
          <Form.Item
            name="name"
            label="类型名称"
            rules={[{ required: true, message: "请输入类型名称" }]}
          >
            <Input placeholder="请输入类型名称" />
          </Form.Item>
          <Form.Item
            name="code"
            label="类型编码"
            rules={[{ required: true, message: "请输入类型编码" }]}
          >
            <Input
              placeholder="请输入类型编码（如 gender）"
              disabled={dialogType === "edit"}
            />
          </Form.Item>
          <Form.Item name="remark" label="备注">
            <Input.TextArea placeholder="请输入备注" rows={2} />
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

DictTypeFormDialog.displayName = "DictTypeFormDialog";
export default DictTypeFormDialog;
