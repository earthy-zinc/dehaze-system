import { UserAPI } from "dehaze-sdk-js";
import { Form, Input, Modal, message } from "antd";
import React, { useCallback, useState } from "react";

interface PasswordResetDialogProps {
  onSuccess?: () => void;
}

export interface PasswordResetDialogRef {
  open: (userId: number, username: string) => void;
}

const PasswordResetDialog = React.forwardRef<
  PasswordResetDialogRef,
  PasswordResetDialogProps
>(({ onSuccess }, ref) => {
  const [visible, setVisible] = useState(false);
  const [confirmLoading, setConfirmLoading] = useState(false);
  const [userId, setUserId] = useState<number>(0);
  const [username, setUsername] = useState("");
  const [form] = Form.useForm();

  const open = useCallback(
    (id: number, name: string) => {
      setUserId(id);
      setUsername(name);
      form.resetFields();
      setVisible(true);
    },
    [form]
  );

  React.useImperativeHandle(ref, () => ({ open }), [open]);

  const handleCancel = useCallback(() => {
    setVisible(false);
    form.resetFields();
  }, [form]);

  const handleSubmit = useCallback(async () => {
    try {
      const values = await form.validateFields();
      setConfirmLoading(true);
      await UserAPI.updatePassword(userId, values.password);
      message.success(`用户「${username}」密码重置成功`);
      handleCancel();
      onSuccess?.();
    } catch (error: any) {
      if (error?.errorFields) return;
      message.error(error?.message || "密码重置失败");
    } finally {
      setConfirmLoading(false);
    }
  }, [form, userId, username, handleCancel, onSuccess]);

  return (
    <Modal
      title="重置密码"
      open={visible}
      confirmLoading={confirmLoading}
      okText="确定"
      cancelText="取消"
      destroyOnClose
      onOk={handleSubmit}
      onCancel={handleCancel}
    >
      <p style={{ marginBottom: 16 }}>
        为用户「<strong>{username}</strong>」重置密码：
      </p>
      <Form form={form} layout="vertical" validateTrigger="onBlur">
        <Form.Item
          name="password"
          label="新密码"
          rules={[
            { required: true, message: "请输入新密码" },
            { min: 6, message: "密码长度不能少于6位" },
          ]}
        >
          <Input.Password placeholder="请输入新密码" />
        </Form.Item>
        <Form.Item
          name="confirmPassword"
          label="确认密码"
          dependencies={["password"]}
          rules={[
            { required: true, message: "请确认新密码" },
            ({ getFieldValue }) => ({
              validator(_, value) {
                if (!value || getFieldValue("password") === value) {
                  return Promise.resolve();
                }
                return Promise.reject(new Error("两次输入的密码不一致"));
              },
            }),
          ]}
        >
          <Input.Password placeholder="请再次输入新密码" />
        </Form.Item>
      </Form>
    </Modal>
  );
});

PasswordResetDialog.displayName = "PasswordResetDialog";

export default PasswordResetDialog;
