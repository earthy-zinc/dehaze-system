import {
  MemberAPI,
  type MemberPageVO,
  type MemberStatusForm,
} from "dehaze-sdk-js";
import { Form, Input, Modal, message } from "antd";
import React, {
  forwardRef,
  useCallback,
  useImperativeHandle,
  useState,
} from "react";

export interface FreezeDialogRef {
  open: (record: MemberPageVO) => void;
}

interface FreezeDialogProps {
  onSuccess?: () => void;
}

const FreezeDialog = forwardRef<FreezeDialogRef, FreezeDialogProps>(
  ({ onSuccess }, ref) => {
    const [visible, setVisible] = useState(false);
    const [confirmLoading, setConfirmLoading] = useState(false);
    const [form] = Form.useForm<MemberStatusForm>();
    const [userId, setUserId] = useState(0);
    const [username, setUsername] = useState("");
    const [status, setStatus] = useState<0 | 1>(0);

    const open = useCallback(
      (record: MemberPageVO) => {
        setUserId(record.userId);
        setUsername(record.username);
        const nextStatus: 0 | 1 = record.status === 1 ? 0 : 1;
        setStatus(nextStatus);
        setVisible(true);
        form.resetFields();
        form.setFieldsValue({ status: nextStatus, reason: "" });
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
        if (status === 0) {
          await form.validateFields();
        }
        const values = form.getFieldsValue();
        setConfirmLoading(true);
        await MemberAPI.updateStatus(userId, {
          status,
          reason: status === 0 ? values.reason : undefined,
        });
        message.success(status === 0 ? "冻结成功" : "解冻成功");
        handleCancel();
        onSuccess?.();
      } catch (error: any) {
        if (error?.errorFields) return;
        message.error(error?.message || "操作失败");
      } finally {
        setConfirmLoading(false);
      }
    }, [status, form, userId, handleCancel, onSuccess]);

    return (
      <Modal
        title={status === 0 ? "冻结会员" : "解冻会员"}
        open={visible}
        width={560}
        confirmLoading={confirmLoading}
        okText="确定"
        cancelText="取消"
        destroyOnHidden
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
          <Form.Item label="会员">
            <span>{username}</span>
          </Form.Item>
          {status === 0 ? (
            <Form.Item
              name="reason"
              label="冻结原因"
              rules={[{ required: true, message: "请输入冻结原因" }]}
            >
              <Input.TextArea
                rows={3}
                maxLength={200}
                showCount
                placeholder="请输入冻结原因"
              />
            </Form.Item>
          ) : (
            <Form.Item label="说明">
              <span>解冻后会员可正常使用所有权益</span>
            </Form.Item>
          )}
        </Form>
      </Modal>
    );
  }
);

FreezeDialog.displayName = "FreezeDialog";

export default FreezeDialog;
