import {
  MemberAPI,
  type MemberLevelAdjustForm,
  type MemberLevelCode,
  type MemberPageVO,
} from "dehaze-sdk-js";
import { DatePicker, Form, Input, Modal, Select, message } from "antd";
import React, {
  forwardRef,
  useCallback,
  useImperativeHandle,
  useState,
} from "react";

const LEVEL_OPTIONS: { label: string; value: MemberLevelCode }[] = [
  { label: "普通会员", value: "level_0" },
  { label: "高级会员", value: "level_1" },
  { label: "VIP会员", value: "level_2" },
  { label: "SVIP会员", value: "level_3" },
];

const LEVEL_ORDER: Record<MemberLevelCode, number> = {
  level_0: 0,
  level_1: 1,
  level_2: 2,
  level_3: 3,
};

export interface LevelAdjustDialogRef {
  open: (record: MemberPageVO) => void;
}

interface LevelAdjustDialogProps {
  onSuccess?: () => void;
}

const LevelAdjustDialog = forwardRef<
  LevelAdjustDialogRef,
  LevelAdjustDialogProps
>(({ onSuccess }, ref) => {
  const [visible, setVisible] = useState(false);
  const [confirmLoading, setConfirmLoading] = useState(false);
  const [form] = Form.useForm<MemberLevelAdjustForm>();
  const [userId, setUserId] = useState(0);
  const [username, setUsername] = useState("");
  const [currentLevelCode, setCurrentLevelCode] =
    useState<MemberLevelCode>("level_0");
  const [currentLevelName, setCurrentLevelName] = useState("");

  const open = useCallback(
    (record: MemberPageVO) => {
      setUserId(record.userId);
      setUsername(record.username);
      setCurrentLevelCode(record.levelCode);
      setCurrentLevelName(record.levelName);
      setVisible(true);
      form.resetFields();
      form.setFieldsValue({
        levelCode: record.levelCode,
        expireTime: undefined,
        reason: "",
      });
    },
    [form]
  );

  useImperativeHandle(ref, () => ({ open }), [open]);

  const handleCancel = useCallback(() => {
    setVisible(false);
    form.resetFields();
  }, [form]);

  const doSubmit = useCallback(
    (values: MemberLevelAdjustForm) => {
      setConfirmLoading(true);
      MemberAPI.adjustLevel(userId, values)
        .then(() => {
          message.success("等级调整成功");
          handleCancel();
          onSuccess?.();
        })
        .catch((error) => {
          message.error(error?.message || "操作失败");
        })
        .finally(() => {
          setConfirmLoading(false);
        });
    },
    [userId, handleCancel, onSuccess]
  );

  const handleSubmit = useCallback(async () => {
    try {
      const values = await form.validateFields();
      const isDowngrade =
        LEVEL_ORDER[currentLevelCode] > LEVEL_ORDER[values.levelCode];
      if (isDowngrade) {
        const targetLabel = LEVEL_OPTIONS.find(
          (o) => o.value === values.levelCode
        )?.label;
        Modal.confirm({
          title: "降级确认",
          content: `确认将会员「${username}」从 ${currentLevelName} 降级为 ${targetLabel} 吗？`,
          okText: "确定",
          cancelText: "取消",
          onOk: () => doSubmit(values),
        });
      } else {
        doSubmit(values);
      }
    } catch (error: any) {
      if (error?.errorFields) return;
      message.error(error?.message || "操作失败");
    }
  }, [form, currentLevelCode, username, currentLevelName, doSubmit]);

  return (
    <Modal
      title="等级调整"
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
          <span>
            {username}（当前：{currentLevelName}）
          </span>
        </Form.Item>
        <Form.Item
          name="levelCode"
          label="目标等级"
          rules={[{ required: true, message: "请选择目标等级" }]}
        >
          <Select
            placeholder="请选择等级"
            options={LEVEL_OPTIONS}
            style={{ width: "100%" }}
          />
        </Form.Item>
        <Form.Item name="expireTime" label="到期时间">
          <DatePicker
            style={{ width: "100%" }}
            placeholder="不选则由成长值维持"
          />
        </Form.Item>
        <Form.Item
          name="reason"
          label="调整原因"
          rules={[
            { required: true, message: "请输入调整原因" },
            { min: 2, max: 200, message: "原因长度为2-200字符" },
          ]}
        >
          <Input.TextArea
            rows={3}
            maxLength={200}
            showCount
            placeholder="请输入2-200字符的调整原因"
          />
        </Form.Item>
      </Form>
    </Modal>
  );
});

LevelAdjustDialog.displayName = "LevelAdjustDialog";

export default LevelAdjustDialog;
