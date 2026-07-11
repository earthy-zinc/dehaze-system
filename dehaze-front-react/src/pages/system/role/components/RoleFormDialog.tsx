import { RoleAPI, type RoleForm } from "dehaze-sdk-js";
import {
  Form,
  Input,
  InputNumber,
  Modal,
  Radio,
  Select,
  message,
} from "antd";
import React, {
  forwardRef,
  useCallback,
  useImperativeHandle,
  useState,
} from "react";

/** 数据权限选项 */
const DATA_SCOPE_OPTIONS = [
  { value: 0, label: "全部数据" },
  { value: 1, label: "部门及子部门数据" },
  { value: 2, label: "本部门数据" },
  { value: 3, label: "本人数据" },
];

export interface RoleFormDialogRef {
  open: (type: "add" | "edit", roleId?: number) => void;
}

interface RoleFormDialogProps {
  onSuccess?: () => void;
}

const RoleFormDialog = forwardRef<RoleFormDialogRef, RoleFormDialogProps>(
  ({ onSuccess }, ref) => {
    const [visible, setVisible] = useState(false);
    const [dialogType, setDialogType] = useState<"add" | "edit">("add");
    const [confirmLoading, setConfirmLoading] = useState(false);
    const [form] = Form.useForm<RoleForm>();

    const open = useCallback(
      async (type: "add" | "edit", roleId?: number) => {
        setDialogType(type);
        setVisible(true);

        if (type === "add") {
          form.resetFields();
          form.setFieldsValue({ sort: 1, status: 1, dataScope: 2 });
        } else if (type === "edit" && roleId) {
          form.resetFields();
          try {
            const data = await RoleAPI.getFormData(roleId);
            form.setFieldsValue({
              id: data.id ?? roleId,
              name: data.name,
              code: data.code,
              dataScope: data.dataScope ?? 2,
              sort: data.sort ?? 1,
              status: data.status ?? 1,
            });
          } catch {
            message.error("获取角色信息失败");
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

        if (dialogType === "edit") {
          const roleId = form.getFieldValue("id");
          await RoleAPI.update(roleId, values);
          message.success("修改角色成功");
        } else {
          await RoleAPI.add(values);
          message.success("新增角色成功");
        }

        handleCancel();
        onSuccess?.();
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
      } catch (error: any) {
        if (error?.errorFields) return;
        message.error(error?.message || "操作失败");
      } finally {
        setConfirmLoading(false);
      }
    }, [form, dialogType, handleCancel, onSuccess]);

    return (
      <Modal
        title={dialogType === "add" ? "新增角色" : "修改角色"}
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
          layout="horizontal"
          labelCol={{ span: 6 }}
          wrapperCol={{ span: 16 }}
          colon={false}
          validateTrigger="onBlur"
        >
          <Form.Item
            name="name"
            label="角色名称"
            rules={[
              { required: true, message: "请输入角色名称" },
              { max: 30, message: "角色名称不能超过30个字符" },
            ]}
          >
            <Input placeholder="请输入角色名称" />
          </Form.Item>

          <Form.Item
            name="code"
            label="角色编码"
            rules={[
              { required: true, message: "请输入角色编码" },
              { pattern: /^[A-Z_]+$/, message: "编码只能包含大写字母和下划线" },
            ]}
          >
            <Input
              placeholder="请输入角色编码（如 ROLE_ADMIN）"
              disabled={dialogType === "edit"}
            />
          </Form.Item>

          <Form.Item
            name="dataScope"
            label="数据权限"
            rules={[{ required: true, message: "请选择数据权限" }]}
          >
            <Select
              placeholder="请选择数据权限"
              options={DATA_SCOPE_OPTIONS}
            />
          </Form.Item>

          <Form.Item
            name="sort"
            label="排序"
            rules={[{ required: true, message: "请输入排序值" }]}
          >
            <InputNumber min={1} style={{ width: "100%" }} placeholder="请输入排序值" />
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

RoleFormDialog.displayName = "RoleFormDialog";

export default RoleFormDialog;
