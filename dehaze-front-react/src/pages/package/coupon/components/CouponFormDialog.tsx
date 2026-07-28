import {
  CouponAPI,
  PackageAPI,
  type CouponForm,
  type CouponType,
  type PackagePageVO,
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
  useEffect,
  useImperativeHandle,
  useState,
} from "react";

const FORM_TIME_FMT = "YYYY-MM-DD HH:mm:ss";

const TYPE_OPTIONS: { label: string; value: CouponType }[] = [
  { label: "满减券", value: "full_reduction" },
  { label: "折扣券", value: "discount" },
  { label: "无门槛券", value: "no_threshold" },
  { label: "体验券", value: "trial" },
];

const DEFAULT_FORM: CouponForm = {
  name: "",
  type: "full_reduction",
  faceValue: 0,
  threshold: 0,
  validType: "fixed",
  validStart: undefined,
  validEnd: undefined,
  validDays: 7,
  totalQty: 100,
  perUserLimit: 1,
  applicableScope: [],
  status: 1,
};

export interface CouponFormDialogRef {
  open: (type: "add" | "edit", id?: number) => void;
}

interface CouponFormDialogProps {
  onSuccess?: () => void;
}

const CouponFormDialog = forwardRef<CouponFormDialogRef, CouponFormDialogProps>(
  ({ onSuccess }, ref) => {
    const [visible, setVisible] = useState(false);
    const [dialogType, setDialogType] = useState<"add" | "edit">("add");
    const [confirmLoading, setConfirmLoading] = useState(false);
    const [form] = Form.useForm<CouponForm>();
    const [packageOptions, setPackageOptions] = useState<PackagePageVO[]>([]);

    const loadPackageOptions = useCallback(() => {
      PackageAPI.getPage({ pageNum: 1, pageSize: 100 })
        .then((data) => {
          setPackageOptions(data.list || []);
        })
        .catch(() => {
          setPackageOptions([]);
        });
    }, []);

    const open = useCallback(
      async (type: "add" | "edit", id?: number) => {
        setDialogType(type);
        setVisible(true);
        form.resetFields();
        if (packageOptions.length === 0) {
          loadPackageOptions();
        }
        if (type === "add") {
          form.setFieldsValue({ ...DEFAULT_FORM });
        } else if (type === "edit" && id) {
          try {
            const list = await CouponAPI.getPage({
              pageNum: 1,
              pageSize: 100,
            });
            const row = (list.list || []).find((c) => c.id === id);
            if (row) {
              form.setFieldsValue({
                id: row.id,
                name: row.name,
                type: row.type,
                faceValue: row.faceValue,
                threshold: row.threshold,
                validType: row.validType,
                validStart: row.validStart,
                validEnd: row.validEnd,
                validDays: row.validDays,
                totalQty: row.totalQty,
                perUserLimit: row.perUserLimit,
                applicableScope: row.applicableScope ?? [],
                status: row.status,
              });
            }
          } catch {
            message.error("获取优惠券信息失败");
          }
        }
      },
      [form, packageOptions.length, loadPackageOptions]
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
        if (dialogType === "edit") {
          const id = form.getFieldValue("id");
          await CouponAPI.update(id, values);
          message.success("修改优惠券成功");
        } else {
          await CouponAPI.add(values);
          message.success("新增优惠券成功");
        }
        handleCancel();
        onSuccess?.();
      } catch (error: any) {
        if (error?.errorFields) return;
        message.error(error?.message || "操作失败");
      } finally {
        setConfirmLoading(false);
      }
    }, [form, dialogType, handleCancel, onSuccess]);

    const watchType = Form.useWatch("type", form);
    const watchValidType = Form.useWatch("validType", form);

    useEffect(() => {
      if (watchType !== "full_reduction") {
        form.setFieldValue("threshold", 0);
      }
    }, [watchType, form]);

    return (
      <Modal
        title={dialogType === "add" ? "新增优惠券" : "修改优惠券"}
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
            label="名称"
            rules={[
              { required: true, message: "请输入优惠券名称" },
              { max: 50, message: "名称不能超过50个字符" },
            ]}
          >
            <Input placeholder="请输入优惠券名称" />
          </Form.Item>

          <Form.Item
            name="type"
            label="类型"
            rules={[{ required: true, message: "请选择类型" }]}
          >
            <Select placeholder="请选择类型" options={TYPE_OPTIONS} />
          </Form.Item>

          <Form.Item
            name="faceValue"
            label="面值"
            rules={[{ required: true, message: "请输入面值" }]}
          >
            <InputNumber
              min={0}
              precision={2}
              style={{ width: "100%" }}
              placeholder="请输入面值"
            />
          </Form.Item>

          <Form.Item
            name="threshold"
            label="门槛"
            rules={[
              {
                validator: (_, value: number) => {
                  if (
                    watchType === "full_reduction" &&
                    (!value || value <= 0)
                  ) {
                    return Promise.reject(new Error("满减券必须填写门槛"));
                  }
                  return Promise.resolve();
                },
              },
            ]}
          >
            <InputNumber
              min={0}
              precision={2}
              style={{ width: "100%" }}
              placeholder="满减必填"
              disabled={watchType !== "full_reduction"}
            />
          </Form.Item>

          <Form.Item
            name="validType"
            label="有效期类型"
            rules={[{ required: true, message: "请选择有效期类型" }]}
          >
            <Radio.Group>
              <Radio value="fixed">固定日期</Radio>
              <Radio value="relative">相对天数</Radio>
            </Radio.Group>
          </Form.Item>

          {watchValidType === "fixed" ? (
            <>
              <Form.Item
                name="validStart"
                label="生效时间"
                rules={[{ required: true, message: "请选择生效时间" }]}
              >
                <DatePicker
                  showTime
                  format={FORM_TIME_FMT}
                  style={{ width: "100%" }}
                  placeholder="生效时间"
                  value={
                    form.getFieldValue("validStart")
                      ? dayjs(form.getFieldValue("validStart"), FORM_TIME_FMT)
                      : undefined
                  }
                  onChange={(_t, str) =>
                    form.setFieldValue(
                      "validStart",
                      typeof str === "string" && str ? str : undefined
                    )
                  }
                />
              </Form.Item>
              <Form.Item
                name="validEnd"
                label="失效时间"
                rules={[{ required: true, message: "请选择失效时间" }]}
              >
                <DatePicker
                  showTime
                  format={FORM_TIME_FMT}
                  style={{ width: "100%" }}
                  placeholder="失效时间"
                  value={
                    form.getFieldValue("validEnd")
                      ? dayjs(form.getFieldValue("validEnd"), FORM_TIME_FMT)
                      : undefined
                  }
                  onChange={(_t, str) =>
                    form.setFieldValue(
                      "validEnd",
                      typeof str === "string" && str ? str : undefined
                    )
                  }
                />
              </Form.Item>
            </>
          ) : (
            <Form.Item
              name="validDays"
              label="有效天数"
              rules={[{ required: true, message: "请输入有效天数" }]}
            >
              <InputNumber
                min={1}
                style={{ width: "100%" }}
                placeholder="请输入有效天数"
              />
            </Form.Item>
          )}

          <Form.Item
            name="totalQty"
            label="总量"
            rules={[{ required: true, message: "请输入总量" }]}
          >
            <InputNumber
              min={0}
              style={{ width: "100%" }}
              placeholder="请输入总量"
            />
          </Form.Item>

          <Form.Item
            name="perUserLimit"
            label="每人限领"
            rules={[{ required: true, message: "请输入每人限领数" }]}
          >
            <InputNumber
              min={1}
              style={{ width: "100%" }}
              placeholder="请输入每人限领数"
            />
          </Form.Item>

          <Form.Item
            name="applicableScope"
            label="适用套餐"
            wrapperCol={{ span: 16 }}
          >
            <Select
              mode="multiple"
              allowClear
              placeholder="不选则全部套餐适用"
              options={packageOptions.map((p) => ({
                label: p.name,
                value: p.id,
              }))}
            />
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
  }
);

CouponFormDialog.displayName = "CouponFormDialog";

export default CouponFormDialog;
