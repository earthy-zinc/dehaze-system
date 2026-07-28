import { MessageTemplateAPI, type MessageTemplateForm } from "dehaze-sdk-js";
import { Checkbox, Form, Input, Modal, Radio, message } from "antd";
import React, {
  forwardRef,
  useCallback,
  useImperativeHandle,
  useState,
} from "react";

const CHANNEL_OPTIONS = [
  { label: "站内信", value: "inbox" },
  { label: "APP 推送", value: "push" },
  { label: "邮件", value: "email" },
];

interface FormValues extends MessageTemplateForm {
  channelsList?: string[];
}

export interface MessageTemplateFormDialogRef {
  open: (id: number) => void;
}

interface MessageTemplateFormDialogProps {
  onSuccess?: () => void;
}

const MessageTemplateFormDialog = forwardRef<
  MessageTemplateFormDialogRef,
  MessageTemplateFormDialogProps
>(({ onSuccess }, ref) => {
  const [visible, setVisible] = useState(false);
  const [editId, setEditId] = useState<number | undefined>(undefined);
  const [confirmLoading, setConfirmLoading] = useState(false);
  const [form] = Form.useForm<FormValues>();

  const open = useCallback(
    async (id: number) => {
      setEditId(id);
      setVisible(true);
      form.resetFields();
      try {
        const data = await MessageTemplateAPI.getDetail(id);
        const channelsList = data.channels
          ? Object.entries(data.channels)
              .filter(([, v]) => v)
              .map(([k]) => k)
          : [];
        form.setFieldsValue({
          name: data.name,
          titleTemplate: data.titleTemplate,
          contentTemplate: data.contentTemplate,
          priority: data.priority,
          status: data.status,
          channelsList,
        });
      } catch {
        message.error("获取模板信息失败");
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

      const channelsMap: Record<string, boolean> = {
        inbox: (values.channelsList ?? []).includes("inbox"),
        push: (values.channelsList ?? []).includes("push"),
        email: (values.channelsList ?? []).includes("email"),
      };

      const payload: MessageTemplateForm = {
        name: values.name,
        titleTemplate: values.titleTemplate,
        contentTemplate: values.contentTemplate,
        priority: values.priority,
        status: values.status,
        channels: channelsMap,
      };

      await MessageTemplateAPI.update(editId!, payload);
      message.success("保存成功");
      handleCancel();
      onSuccess?.();
    } catch (error: any) {
      if (error?.errorFields) return;
      message.error(error?.message || "操作失败");
    } finally {
      setConfirmLoading(false);
    }
  }, [form, editId, handleCancel, onSuccess]);

  return (
    <Modal
      title="编辑模板"
      open={visible}
      width={680}
      confirmLoading={confirmLoading}
      okText="保存"
      cancelText="取消"
      destroyOnHidden
      onOk={handleSubmit}
      onCancel={handleCancel}
    >
      <Form
        form={form}
        layout="horizontal"
        labelCol={{ span: 5 }}
        wrapperCol={{ span: 17 }}
        colon={false}
        validateTrigger="onBlur"
        initialValues={{ priority: 2, status: 1, channelsList: ["inbox"] }}
      >
        <Form.Item
          name="name"
          label="模板名称"
          rules={[{ required: true, message: "请输入模板名称" }]}
        >
          <Input placeholder="请输入模板名称" />
        </Form.Item>

        <Form.Item
          name="titleTemplate"
          label="标题模板"
          rules={[{ required: true, message: "请输入标题模板" }]}
        >
          <Input.TextArea
            rows={2}
            placeholder="支持变量占位，如：恭喜您升级至 {levelName}"
          />
        </Form.Item>

        <Form.Item
          name="contentTemplate"
          label="正文模板"
          rules={[{ required: true, message: "请输入正文模板" }]}
        >
          <Input.TextArea
            rows={6}
            placeholder="支持变量占位，如：您已解锁以下新权益：{benefitList}"
          />
        </Form.Item>

        <Form.Item name="priority" label="默认优先级">
          <Radio.Group>
            <Radio value={1}>低</Radio>
            <Radio value={2}>中</Radio>
            <Radio value={3}>高</Radio>
            <Radio value={4}>紧急</Radio>
          </Radio.Group>
        </Form.Item>

        <Form.Item name="channelsList" label="推送渠道">
          <Checkbox.Group options={CHANNEL_OPTIONS} />
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
});

MessageTemplateFormDialog.displayName = "MessageTemplateFormDialog";

export default MessageTemplateFormDialog;
