import {
  FeedbackAPI,
  type FeedbackCreateForm,
  type FeedbackType,
} from "dehaze-sdk-js";
import { Form, Input, Modal, Select, message } from "antd";
import React, {
  forwardRef,
  useCallback,
  useImperativeHandle,
  useState,
} from "react";

const TYPE_OPTIONS: { value: FeedbackType; label: string }[] = [
  { value: "suggestion", label: "功能建议" },
  { value: "bug", label: "问题报告" },
  { value: "experience", label: "体验反馈" },
  { value: "complaint", label: "投诉" },
];

const MODULE_OPTIONS = [
  { label: "去雾处理", value: "dehaze" },
  { label: "指标评估", value: "evaluate" },
  { label: "数据集", value: "dataset" },
  { label: "会员", value: "member" },
  { label: "套餐", value: "package" },
  { label: "订单", value: "order" },
  { label: "其他", value: "other" },
];

export interface FeedbackFormDialogRef {
  open: () => void;
}

interface FeedbackFormDialogProps {
  onSuccess?: () => void;
}

const FeedbackFormDialog = forwardRef<
  FeedbackFormDialogRef,
  FeedbackFormDialogProps
>(({ onSuccess }, ref) => {
  const [visible, setVisible] = useState(false);
  const [confirmLoading, setConfirmLoading] = useState(false);
  const [form] = Form.useForm<FeedbackCreateForm>();

  const open = useCallback(() => {
    form.resetFields();
    form.setFieldsValue({ feedbackType: "suggestion" });
    setVisible(true);
  }, [form]);

  useImperativeHandle(ref, () => ({ open }), [open]);

  const handleCancel = useCallback(() => {
    setVisible(false);
    form.resetFields();
  }, [form]);

  const handleSubmit = useCallback(async () => {
    try {
      const values = await form.validateFields();
      setConfirmLoading(true);
      await FeedbackAPI.createFeedback(values);
      message.success("反馈提交成功");
      handleCancel();
      onSuccess?.();
    } catch (error: any) {
      if (error?.errorFields) return;
      message.error(error?.message || "提交失败");
    } finally {
      setConfirmLoading(false);
    }
  }, [form, handleCancel, onSuccess]);

  return (
    <Modal
      title="新建反馈"
      open={visible}
      width={560}
      confirmLoading={confirmLoading}
      okText="提交"
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
          name="feedbackType"
          label="反馈类型"
          rules={[{ required: true, message: "请选择反馈类型" }]}
        >
          <Select placeholder="请选择反馈类型" options={TYPE_OPTIONS} />
        </Form.Item>
        <Form.Item
          name="title"
          label="标题"
          rules={[
            { required: true, message: "请输入标题" },
            { min: 5, max: 50, message: "标题长度为 5-50 字符" },
          ]}
        >
          <Input
            maxLength={50}
            showCount
            placeholder="请输入标题（5-50 字符）"
          />
        </Form.Item>
        <Form.Item
          name="content"
          label="内容"
          rules={[
            { required: true, message: "请输入内容" },
            { min: 10, max: 1000, message: "内容长度为 10-1000 字符" },
          ]}
        >
          <Input.TextArea
            rows={6}
            maxLength={1000}
            showCount
            placeholder="请详细描述您的问题或建议（10-1000 字符）"
          />
        </Form.Item>
        <Form.Item name="relatedModule" label="相关模块">
          <Select
            allowClear
            placeholder="请选择相关模块"
            options={MODULE_OPTIONS}
          />
        </Form.Item>
        <Form.Item name="contact" label="联系方式">
          <Input placeholder="手机/邮箱（仅管理员可见）" />
        </Form.Item>
      </Form>
    </Modal>
  );
});

FeedbackFormDialog.displayName = "FeedbackFormDialog";

export default FeedbackFormDialog;
