import { FeedbackAPI } from "dehaze-sdk-js";
import { Form, Modal, Select, message } from "antd";
import React, {
  forwardRef,
  useCallback,
  useImperativeHandle,
  useState,
} from "react";

export interface FeedbackTagDialogRef {
  open: (feedbackId: number, currentTags?: string[]) => void;
}

interface FeedbackTagDialogProps {
  onSuccess?: () => void;
}

const PRESET_TAGS = [
  "高优先级",
  "已确认",
  "待验证",
  "重复反馈",
  "已知问题",
  "新需求",
  "性能问题",
  "兼容性",
];

const FeedbackTagDialog = forwardRef<
  FeedbackTagDialogRef,
  FeedbackTagDialogProps
>(({ onSuccess }, ref) => {
  const [visible, setVisible] = useState(false);
  const [confirmLoading, setConfirmLoading] = useState(false);
  const [feedbackId, setFeedbackId] = useState<number>(0);
  const [tags, setTags] = useState<string[]>([]);

  const open = useCallback((id: number, currentTags?: string[]) => {
    setFeedbackId(id);
    setTags(currentTags ? [...currentTags] : []);
    setVisible(true);
  }, []);

  useImperativeHandle(ref, () => ({ open }), [open]);

  const handleCancel = useCallback(() => {
    setVisible(false);
    setTags([]);
  }, []);

  const handleSubmit = useCallback(async () => {
    setConfirmLoading(true);
    try {
      await FeedbackAPI.updateFeedbackTags(feedbackId, tags);
      message.success("标签更新成功");
      handleCancel();
      onSuccess?.();
    } catch (error: any) {
      message.error(error?.message || "操作失败");
    } finally {
      setConfirmLoading(false);
    }
  }, [feedbackId, tags, handleCancel, onSuccess]);

  return (
    <Modal
      title="编辑标签"
      open={visible}
      width={500}
      confirmLoading={confirmLoading}
      okText="确定"
      cancelText="取消"
      destroyOnHidden
      onOk={handleSubmit}
      onCancel={handleCancel}
    >
      <Form
        layout="horizontal"
        labelCol={{ span: 4 }}
        wrapperCol={{ span: 20 }}
      >
        <Form.Item label="标签">
          <Select
            mode="tags"
            value={tags}
            onChange={setTags}
            placeholder="输入标签后回车，或从下拉选项选择"
            style={{ width: "100%" }}
            options={PRESET_TAGS.map((t) => ({ value: t, label: t }))}
            tokenSeparators={[","]}
          />
        </Form.Item>
      </Form>
    </Modal>
  );
});

FeedbackTagDialog.displayName = "FeedbackTagDialog";

export default FeedbackTagDialog;
