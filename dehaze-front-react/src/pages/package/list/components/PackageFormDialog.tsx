import {
  PackageAPI,
  type PackageForm,
} from "dehaze-sdk-js";
import {
  Divider,
  Form,
  Input,
  InputNumber,
  Modal,
  Radio,
  Select,
  message,
} from "antd";
import React, {
  forwardRef,
  useCallback,
  useImperativeHandle,
  useState,
} from "react";

const LEVEL_OPTIONS = [
  { value: "level_1", label: "基础版" },
  { value: "level_2", label: "专业版" },
  { value: "level_3", label: "旗舰版" },
];

const PERIOD_OPTIONS = [
  { value: "monthly", label: "月卡" },
  { value: "quarterly", label: "季卡" },
  { value: "yearly", label: "年卡" },
];

const PERIOD_DAYS_MAP: Record<string, number> = {
  monthly: 30,
  quarterly: 90,
  yearly: 365,
};

const DEFAULT_FORM: PackageForm = {
  name: "",
  levelCode: "level_1",
  period: "monthly",
  periodDays: 30,
  originalPrice: 0,
  salePrice: 0,
  description: "",
  sort: 1,
  status: 1,
  benefitOverrides: {
    monthlyDehazeQuota: 0,
    monthlyEvaluateQuota: 0,
    historyRetention: 0,
    batchLimit: 0,
    priority: 0,
    advancedParams: 0,
    hdExport: 0,
    reportExport: 0,
    batchDownload: 0,
  },
};

export interface PackageFormDialogRef {
  open: (type: "add" | "edit", id?: number) => void;
}

interface PackageFormDialogProps {
  onSuccess?: () => void;
}

const PackageFormDialog = forwardRef<
  PackageFormDialogRef,
  PackageFormDialogProps
>(({ onSuccess }, ref) => {
  const [visible, setVisible] = useState(false);
  const [dialogType, setDialogType] = useState<"add" | "edit">("add");
  const [confirmLoading, setConfirmLoading] = useState(false);
  const [form] = Form.useForm<PackageForm>();

  const open = useCallback(
    async (type: "add" | "edit", id?: number) => {
      setDialogType(type);
      setVisible(true);
      form.resetFields();
      if (type === "add") {
        form.setFieldsValue({ ...DEFAULT_FORM });
      } else if (type === "edit" && id) {
        try {
          const data = await PackageAPI.getForm(id);
          form.setFieldsValue({
            id: data.id ?? id,
            name: data.name,
            levelCode: data.levelCode,
            period: data.period,
            periodDays: data.periodDays,
            originalPrice: data.originalPrice,
            salePrice: data.salePrice,
            description: data.description,
            sort: data.sort ?? 1,
            status: data.status ?? 1,
            benefitOverrides: {
              monthlyDehazeQuota: 0,
              monthlyEvaluateQuota: 0,
              historyRetention: 0,
              batchLimit: 0,
              priority: 0,
              advancedParams: 0,
              hdExport: 0,
              reportExport: 0,
              batchDownload: 0,
              ...(data.benefitOverrides || {}),
            },
          });
        } catch {
          message.error("获取套餐信息失败");
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

  const handlePeriodChange = useCallback(
    (value: string) => {
      form.setFieldValue("periodDays", PERIOD_DAYS_MAP[value] ?? 30);
    },
    [form]
  );

  const handleSubmit = useCallback(async () => {
    try {
      const values = await form.validateFields();
      setConfirmLoading(true);
      if (dialogType === "edit") {
        const id = form.getFieldValue("id");
        await PackageAPI.update(id, values);
        message.success("修改套餐成功");
      } else {
        await PackageAPI.add(values);
        message.success("新增套餐成功");
      }
      handleCancel();
      onSuccess?.();
    } catch (error: unknown) {
      if (error && typeof error === "object" && "errorFields" in error) return;
      message.error(
        error instanceof Error ? error.message : "操作失败"
      );
    } finally {
      setConfirmLoading(false);
    }
  }, [form, dialogType, handleCancel, onSuccess]);

  return (
    <Modal
      title={dialogType === "add" ? "新增套餐" : "修改套餐"}
      open={visible}
      width={720}
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
        labelCol={{ span: 8 }}
        wrapperCol={{ span: 16 }}
        colon={false}
        validateTrigger="onBlur"
      >
        <Form.Item
          name="name"
          label="套餐名"
          rules={[
            { required: true, message: "请输入套餐名" },
            { max: 50, message: "套餐名不能超过50个字符" },
          ]}
        >
          <Input placeholder="请输入套餐名" />
        </Form.Item>

        <Form.Item
          name="levelCode"
          label="等级"
          rules={[{ required: true, message: "请选择等级" }]}
        >
          <Select placeholder="请选择等级" options={LEVEL_OPTIONS} />
        </Form.Item>

        <Form.Item
          name="period"
          label="计费周期"
          rules={[{ required: true, message: "请选择计费周期" }]}
        >
          <Select
            placeholder="请选择计费周期"
            options={PERIOD_OPTIONS}
            onChange={handlePeriodChange}
          />
        </Form.Item>

        <Form.Item
          name="periodDays"
          label="周期天数"
          rules={[{ required: true, message: "请输入周期天数" }]}
        >
          <InputNumber
            min={1}
            style={{ width: "100%" }}
            placeholder="请输入周期天数"
          />
        </Form.Item>

        <Form.Item
          name="originalPrice"
          label="原价"
          rules={[{ required: true, message: "请输入原价" }]}
        >
          <InputNumber
            min={0}
            precision={2}
            style={{ width: "100%" }}
            placeholder="请输入原价"
            prefix="¥"
          />
        </Form.Item>

        <Form.Item
          name="salePrice"
          label="售价"
          rules={[{ required: true, message: "请输入售价" }]}
        >
          <InputNumber
            min={0}
            precision={2}
            style={{ width: "100%" }}
            placeholder="请输入售价"
            prefix="¥"
          />
        </Form.Item>

        <Form.Item name="sort" label="排序号">
          <InputNumber
            min={0}
            style={{ width: "100%" }}
            placeholder="请输入排序号"
          />
        </Form.Item>

        <Form.Item name="status" label="状态">
          <Radio.Group>
            <Radio value={1}>在售</Radio>
            <Radio value={0}>下架</Radio>
          </Radio.Group>
        </Form.Item>

        <Form.Item name="description" label="描述" wrapperCol={{ span: 16 }}>
          <Input.TextArea rows={2} placeholder="套餐描述" />
        </Form.Item>

        <Divider orientation="left" plain>
          权益覆盖配置
        </Divider>

        <Form.Item
          name={["benefitOverrides", "monthlyDehazeQuota"]}
          label="去雾配额"
        >
          <InputNumber min={0} style={{ width: "100%" }} />
        </Form.Item>
        <Form.Item
          name={["benefitOverrides", "monthlyEvaluateQuota"]}
          label="评估配额"
        >
          <InputNumber min={0} style={{ width: "100%" }} />
        </Form.Item>
        <Form.Item
          name={["benefitOverrides", "historyRetention"]}
          label="历史保留(天)"
        >
          <InputNumber min={0} style={{ width: "100%" }} />
        </Form.Item>
        <Form.Item
          name={["benefitOverrides", "batchLimit"]}
          label="批量上限"
        >
          <InputNumber min={0} style={{ width: "100%" }} />
        </Form.Item>
        <Form.Item
          name={["benefitOverrides", "priority"]}
          label="优先级"
        >
          <InputNumber min={0} style={{ width: "100%" }} />
        </Form.Item>
        <Form.Item
          name={["benefitOverrides", "advancedParams"]}
          label="高级参数"
        >
          <InputNumber min={0} style={{ width: "100%" }} />
        </Form.Item>
        <Form.Item
          name={["benefitOverrides", "hdExport"]}
          label="高清导出"
        >
          <InputNumber min={0} style={{ width: "100%" }} />
        </Form.Item>
        <Form.Item
          name={["benefitOverrides", "reportExport"]}
          label="报告导出"
        >
          <InputNumber min={0} style={{ width: "100%" }} />
        </Form.Item>
        <Form.Item
          name={["benefitOverrides", "batchDownload"]}
          label="批量下载"
        >
          <InputNumber min={0} style={{ width: "100%" }} />
        </Form.Item>
      </Form>
    </Modal>
  );
});

PackageFormDialog.displayName = "PackageFormDialog";

export default PackageFormDialog;
