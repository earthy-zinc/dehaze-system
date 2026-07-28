import {
  MemberAPI,
  type MemberGrowthAdjustForm,
  type MemberPageVO,
} from "dehaze-sdk-js";
import { Form, Input, InputNumber, Modal, message } from "antd";
import React, {
  forwardRef,
  useCallback,
  useImperativeHandle,
  useState,
} from "react";

export interface GrowthAdjustDialogRef {
  open: (record: MemberPageVO) => void;
}

interface GrowthAdjustDialogProps {
  onSuccess?: () => void;
}

const GrowthAdjustDialog = forwardRef<
  GrowthAdjustDialogRef,
  GrowthAdjustDialogProps
>(({ onSuccess }, ref) => {
  const [visible, setVisible] = useState(false);
  const [confirmLoading, setConfirmLoading] = useState(false);
  const [form] = Form.useForm<MemberGrowthAdjustForm>();
  const [userId, setUserId] = useState(0);
  const [username, setUsername] = useState("");
  const [currentGrowth, setCurrentGrowth] = useState(0);
  const [changeValue, setChangeValue] = useState(0);

  const expectedGrowth = currentGrowth + (changeValue || 0);

  const open = useCallback(
    (record: MemberPageVO) => {
      setUserId(record.userId);
      setUsername(record.username);
      setCurrentGrowth(record.growthValue);
      setChangeValue(0);
      setVisible(true);
      form.resetFields();
      form.setFieldsValue({ changeValue: 0, reason: "" });
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
      await MemberAPI.adjustGrowth(userId, values);
      message.success("成长值调整成功");
      handleCancel();
      onSuccess?.();
    } catch (error: any) {
      if (error?.errorFields) return;
      message.error(error?.message || "操作失败");
    } finally {
      setConfirmLoading(false);
    }
  }, [form, userId, handleCancel, onSuccess]);

  return (
    <Modal
      title="成长值调整"
      open={visible}
      width={560}
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
        labelCol={{ span: 6 }}
        wrapperCol={{ span: 16 }}
        colon={false}
        validateTrigger="onBlur"
      >
        <Form.Item label="会员">
          <span>{username}</span>
        </Form.Item>
        <Form.Item label="当前成长值">
          <span>{currentGrowth}</span>
        </Form.Item>
        <Form.Item
          name="changeValue"
          label="变动值"
          rules={[{ required: true, message: "请输入变动值" }]}
        >
          <InputNumber
            precision={0}
            step={1}
            style={{ width: 200 }}
            onChange={(v) => setChangeValue(typeof v === "number" ? v : 0)}
          />
        </Form.Item>
        <Form.Item label="预览">
          <span>
            {currentGrowth} + {changeValue || 0} ={" "}
            <strong>{expectedGrowth}</strong>
          </span>
        </Form.Item>
        <Form.Item
          name="reason"
          label="调整原因"
          rules={[{ required: true, message: "请输入调整原因" }]}
        >
          <Input.TextArea
            rows={3}
            maxLength={200}
            showCount
            placeholder="请输入调整原因"
          />
        </Form.Item>
      </Form>
    </Modal>
  );
});

GrowthAdjustDialog.displayName = "GrowthAdjustDialog";

export default GrowthAdjustDialog;
