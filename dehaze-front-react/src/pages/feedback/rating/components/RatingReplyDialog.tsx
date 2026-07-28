import { FeedbackAPI } from "dehaze-sdk-js";
import { Form, Input, Modal, message } from "antd";
import React, {
  forwardRef,
  useCallback,
  useImperativeHandle,
  useState,
} from "react";

export interface RatingReplyDialogRef {
  open: (ratingId: number) => void;
}

interface RatingReplyDialogProps {
  onSuccess?: () => void;
}

interface ReplyFormValues {
  content: string;
}

const RatingReplyDialog = forwardRef<
  RatingReplyDialogRef,
  RatingReplyDialogProps
>(({ onSuccess }, ref) => {
  const [visible, setVisible] = useState(false);
  const [confirmLoading, setConfirmLoading] = useState(false);
  const [ratingId, setRatingId] = useState<number>(0);
  const [form] = Form.useForm<ReplyFormValues>();

  const open = useCallback(
    (id: number) => {
      setRatingId(id);
      form.resetFields();
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
      await FeedbackAPI.replyRating(ratingId, values.content);
      message.success("回复成功");
      handleCancel();
      onSuccess?.();
    } catch (error: any) {
      if (error?.errorFields) return;
      message.error(error?.message || "操作失败");
    } finally {
      setConfirmLoading(false);
    }
  }, [form, ratingId, handleCancel, onSuccess]);

  return (
    <Modal
      title="回复评价"
      open={visible}
      width={600}
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

RatingReplyDialog.displayName = "RatingReplyDialog";

export default RatingReplyDialog;
