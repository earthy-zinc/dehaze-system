import {
  FeedbackAPI,
  type FeedbackReplyForm,
  type FeedbackReplyType,
} from "dehaze-sdk-js";
import { Form, Input, Modal, Select, message } from "antd";
import React, {
  forwardRef,
  useCallback,
  useImperativeHandle,
  useState,
} from "react";

const REPLY_TYPE_OPTIONS: { value: FeedbackReplyType; label: string }[] = [
  { value: "info", label: "通知" },
  { value: "resolved", label: "已解决" },
  { value: "unsupported", label: "不支持" },
  { value: "dev_transfer", label: "转开发" },
];

export interface FeedbackReplyDialogRef {
  open: (feedbackId: number) => void;
}

interface FeedbackReplyDialogProps {
  onSuccess?: () => void;
}

const FeedbackReplyDialog = forwardRef<
  FeedbackReplyDialogRef,
  FeedbackReplyDialogProps
>(({ onSuccess }, ref) => {
  const [visible, setVisible] = useState(false);
  const [confirmLoading, setConfirmLoading] = useState(false);
  const [feedbackId, setFeedbackId] = useState<number>(0);
  const [form] = Form.useForm<FeedbackReplyForm>();

  const open = useCallback(
    (id: number) => {
      setFeedbackId(id);
      form.resetFields();
      form.setFieldsValue({ replyType: "info", content: "" });
      setVisible(true);
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
      await FeedbackAPI.replyFeedback(feedbackId, {
        replyType: values.replyType,
        content: values.content,
      });
      message.success("回复成功");
      handleCancel();
      onSuccess?.();
    } catch (error: any) {
      if (error?.errorFields) return;
      message.error(error?.message || "操作失败");
    } finally {
      setConfirmLoading(false);
    }
  }, [form, feedbackId, handleCancel, onSuccess]);

  return (
    <Modal
      title="回复反馈"
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
        labelCol={{ span: 5 }}
        wrapperCol={{ span: 18 }}
        colon={false}
        validateTrigger="onBlur"
      >
        <Form.Item
          name="replyType"
          label="回复类型"
          rules={[{ required: true, message: "请选择回复类型" }]}
        >
          <Select
            placeholder="请选择"
            style={{ width: 200 }}
            options={REPLY_TYPE_OPTIONS}
          />
        </Form.Item>
        <Form.Item
          name="content"
          label="回复内容"
          rules={[
            { required: true, message: "请输入回复内容" },
            {
              min: 10,
              max: 2000,
              message: "回复内容长度为 10-2000 字符",
            },
          ]}
        >
          <Input.TextArea
            rows={4}
            maxLength={2000}
            showCount
            placeholder="请输入回复内容（10-2000 字符）"
          />
        </Form.Item>
      </Form>
    </Modal>
  );
});

FeedbackReplyDialog.displayName = "FeedbackReplyDialog";

export default FeedbackReplyDialog;
