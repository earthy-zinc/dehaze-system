import { FeedbackAPI, type FeedbackAssignForm } from "dehaze-sdk-js";
import { Form, InputNumber, Modal, message } from "antd";
import React, {
  forwardRef,
  useCallback,
  useImperativeHandle,
  useState,
} from "react";

export interface FeedbackAssignDialogRef {
  open: (feedbackId: number, currentAssigneeId?: number) => void;
}

interface FeedbackAssignDialogProps {
  onSuccess?: () => void;
}

const FeedbackAssignDialog = forwardRef<
  FeedbackAssignDialogRef,
  FeedbackAssignDialogProps
>(({ onSuccess }, ref) => {
  const [visible, setVisible] = useState(false);
  const [confirmLoading, setConfirmLoading] = useState(false);
  const [feedbackId, setFeedbackId] = useState<number>(0);
  const [form] = Form.useForm<FeedbackAssignForm>();

  const open = useCallback(
    (id: number, currentAssigneeId?: number) => {
      setFeedbackId(id);
      form.resetFields();
      form.setFieldsValue({ assigneeId: currentAssigneeId ?? 1 });
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
      await FeedbackAPI.assignFeedback(feedbackId, {
        assigneeId: values.assigneeId,
      });
      message.success("分配成功");
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
      title="分配处理人"
      open={visible}
      width={460}
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
        wrapperCol={{ span: 14 }}
        colon={false}
      >
        <Form.Item
          name="assigneeId"
          label="处理人ID"
          rules={[{ required: true, message: "请输入处理人ID" }]}
        >
          <InputNumber min={1} style={{ width: 200 }} />
        </Form.Item>
      </Form>
    </Modal>
  );
});

FeedbackAssignDialog.displayName = "FeedbackAssignDialog";

export default FeedbackAssignDialog;
