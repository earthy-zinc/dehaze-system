import {
  AnnouncementAPI,
  type AnnouncementForm,
  type AnnouncementVO,
} from "dehaze-sdk-js";
import {
  DatePicker,
  Form,
  Input,
  InputNumber,
  Modal,
  Radio,
  Select,
  message,
} from "antd";
import dayjs from "dayjs";
import React, {
  forwardRef,
  useCallback,
  useImperativeHandle,
  useState,
} from "react";

const TYPE_OPTIONS = [
  { value: "maintenance", label: "系统维护" },
  { value: "feature", label: "功能更新" },
  { value: "activity", label: "活动通知" },
  { value: "operation", label: "运营公告" },
];

const TARGET_SCOPE_OPTIONS = [
  { value: "all", label: "全体用户" },
  { value: "level", label: "按会员等级" },
  { value: "specified", label: "指定用户" },
];

const FORM_TIME_FMT = "YYYY-MM-DD HH:mm:ss";

interface FormValues {
  title: string;
  content: string;
  type: string;
  importance: number;
  targetScope: string;
  targetLevel?: number;
  targetUserIdsStr?: string;
  sendTime?: dayjs.Dayjs | null;
  expireTime?: dayjs.Dayjs | null;
}

export interface AnnouncementFormDialogRef {
  open: (type: "add" | "edit", id?: number) => void;
}

interface AnnouncementFormDialogProps {
  onSuccess?: () => void;
}

const AnnouncementFormDialog = forwardRef<
  AnnouncementFormDialogRef,
  AnnouncementFormDialogProps
>(({ onSuccess }, ref) => {
  const [visible, setVisible] = useState(false);
  const [dialogType, setDialogType] = useState<"add" | "edit">("add");
  const [editId, setEditId] = useState<number | undefined>(undefined);
  const [confirmLoading, setConfirmLoading] = useState(false);
  const [form] = Form.useForm<FormValues>();

  const open = useCallback(
    async (type: "add" | "edit", id?: number) => {
      setDialogType(type);
      setVisible(true);

      if (type === "add") {
        form.resetFields();
        form.setFieldsValue({
          type: "maintenance",
          importance: 1,
          targetScope: "all",
          targetLevel: 1,
        });
      } else if (type === "edit" && id) {
        setEditId(id);
        form.resetFields();
        try {
          const data: AnnouncementVO = await AnnouncementAPI.getDetail(id);
          const values: FormValues = {
            title: data.title,
            content: data.content ?? "",
            type: data.type,
            importance: data.importance,
            targetScope: data.targetScope,
            sendTime: data.sendTime
              ? dayjs(data.sendTime, FORM_TIME_FMT)
              : null,
            expireTime: data.expireTime
              ? dayjs(data.expireTime, FORM_TIME_FMT)
              : null,
          };
          if (data.targetParams) {
            if (data.targetScope === "level") {
              values.targetLevel = data.targetParams.level ?? 1;
            } else if (data.targetScope === "specified") {
              values.targetUserIdsStr = (data.targetParams.userIds ?? []).join(
                ","
              );
            }
          }
          form.setFieldsValue(values);
        } catch {
          message.error("获取公告信息失败");
        }
      }
    },
    [form]
  );

  useImperativeHandle(ref, () => ({ open }), [open]);

  const handleCancel = useCallback(() => {
    setVisible(false);
    form.resetFields();
  }, [form]);

  const buildTargetParams = (values: FormValues) => {
    if (values.targetScope === "level") {
      return { level: values.targetLevel ?? 1 };
    }
    if (values.targetScope === "specified") {
      const ids = (values.targetUserIdsStr ?? "")
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean)
        .map(Number)
        .filter((n) => !isNaN(n) && n > 0);
      return { userIds: ids };
    }
    return undefined;
  };

  const handleSubmit = useCallback(async () => {
    try {
      const values = await form.validateFields();
      setConfirmLoading(true);

      const payload: AnnouncementForm = {
        title: values.title,
        content: values.content,
        type: values.type,
        importance: values.importance,
        targetScope: values.targetScope,
        targetParams: buildTargetParams(values),
        sendTime: values.sendTime
          ? values.sendTime.format(FORM_TIME_FMT)
          : undefined,
        expireTime: values.expireTime
          ? values.expireTime.format(FORM_TIME_FMT)
          : undefined,
      };

      if (dialogType === "edit" && editId) {
        await AnnouncementAPI.update(editId, payload);
        message.success("修改成功");
      } else {
        await AnnouncementAPI.create(payload);
        message.success("新增成功");
      }

      handleCancel();
      onSuccess?.();
    } catch (error: unknown) {
      const formErr = error as { errorFields?: Array<{ name: unknown }> };
      if (formErr.errorFields) return;
      const msgErr = error as { message?: string };
      message.error(msgErr.message || "操作失败");
    } finally {
      setConfirmLoading(false);
    }
  }, [form, dialogType, editId, handleCancel, onSuccess]);

  const targetScope = Form.useWatch("targetScope", form);

  return (
    <Modal
      title={dialogType === "add" ? "新增公告" : "编辑公告"}
      open={visible}
      width={640}
      confirmLoading={confirmLoading}
      okText="保存"
      cancelText="取消"
      forceRender
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
      >
        <Form.Item
          name="title"
          label="公告标题"
          rules={[
            { required: true, message: "请输入公告标题" },
            { min: 2, max: 50, message: "标题长度 2-50 字符" },
          ]}
        >
          <Input placeholder="2-50 字符" maxLength={50} showCount />
        </Form.Item>

        <Form.Item
          name="content"
          label="公告内容"
          rules={[{ required: true, message: "请输入公告内容" }]}
        >
          <Input.TextArea rows={5} placeholder="请输入公告内容" />
        </Form.Item>

        <Form.Item
          name="type"
          label="公告类型"
          rules={[{ required: true, message: "请选择公告类型" }]}
        >
          <Select placeholder="请选择" options={TYPE_OPTIONS} />
        </Form.Item>

        <Form.Item
          name="importance"
          label="重要级别"
          rules={[{ required: true, message: "请选择重要级别" }]}
        >
          <Radio.Group>
            <Radio value={1}>普通</Radio>
            <Radio value={2}>重要</Radio>
          </Radio.Group>
        </Form.Item>

        <Form.Item
          name="targetScope"
          label="发送范围"
          rules={[{ required: true, message: "请选择发送范围" }]}
        >
          <Select placeholder="请选择" options={TARGET_SCOPE_OPTIONS} />
        </Form.Item>

        {targetScope === "level" && (
          <Form.Item
            name="targetLevel"
            label="会员等级"
            rules={[{ required: true }]}
          >
            <InputNumber min={1} max={10} />
          </Form.Item>
        )}

        {targetScope === "specified" && (
          <Form.Item
            name="targetUserIdsStr"
            label="用户ID"
            rules={[{ required: true, message: "请输入用户ID" }]}
          >
            <Input placeholder="多个用英文逗号分隔，如 1,2,3" />
          </Form.Item>
        )}

        <Form.Item name="sendTime" label="定时发送">
          <DatePicker
            showTime
            format={FORM_TIME_FMT}
            placeholder="留空则保存为草稿"
            style={{ width: "100%" }}
          />
        </Form.Item>

        <Form.Item name="expireTime" label="过期时间">
          <DatePicker
            showTime
            format={FORM_TIME_FMT}
            placeholder="可选"
            style={{ width: "100%" }}
          />
        </Form.Item>
      </Form>
    </Modal>
  );
});

AnnouncementFormDialog.displayName = "AnnouncementFormDialog";

export default AnnouncementFormDialog;
