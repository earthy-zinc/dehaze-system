import {
  CouponAPI,
  type CouponBatchDistributeForm,
  type CouponVO,
} from "dehaze-sdk-js";
import { Form, Input, Modal, Radio, Select, message } from "antd";
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

type TargetScope = "all" | "level" | "users";

interface DistributeFormValues {
  targetScope: TargetScope;
  levelCodes?: string[];
  userIdsInput?: string;
}

const parseUserIds = (input: string): number[] =>
  input
    .split(/[,，\s]+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map((s) => Number(s))
    .filter((n) => !Number.isNaN(n) && n > 0);

export interface DistributeDialogRef {
  open: (coupon: CouponVO) => void;
}

interface DistributeDialogProps {
  onSuccess?: () => void;
}

const DistributeDialog = forwardRef<DistributeDialogRef, DistributeDialogProps>(
  ({ onSuccess }, ref) => {
    const [visible, setVisible] = useState(false);
    const [confirmLoading, setConfirmLoading] = useState(false);
    const [coupon, setCoupon] = useState<CouponVO | null>(null);
    const [form] = Form.useForm<DistributeFormValues>();

    const open = useCallback(
      (c: CouponVO) => {
        setCoupon(c);
        setVisible(true);
        form.resetFields();
        form.setFieldsValue({ targetScope: "all" });
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
        if (!coupon) return;
        setConfirmLoading(true);
        const payload: CouponBatchDistributeForm = {
          couponId: coupon.id,
          targetScope: values.targetScope,
        };
        if (values.targetScope === "level") {
          payload.levelCodes = values.levelCodes;
        } else if (values.targetScope === "users") {
          payload.userIds = parseUserIds(values.userIdsInput || "");
        }
        const res = await CouponAPI.batchDistribute(payload);
        message.success(
          `发放完成：成功 ${res.successCount} 个，失败 ${res.failCount} 个`
        );
        handleCancel();
        onSuccess?.();
      } catch (error: any) {
        if (error?.errorFields) return;
        message.error(error?.message || "发放失败");
      } finally {
        setConfirmLoading(false);
      }
    }, [form, coupon, handleCancel, onSuccess]);

    const watchTargetScope = Form.useWatch("targetScope", form);

    return (
      <Modal
        title="发放优惠券"
        open={visible}
        width={520}
        confirmLoading={confirmLoading}
        okText="确认发放"
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
        >
          <Form.Item label="优惠券">
            <Input value={coupon?.name || ""} disabled />
          </Form.Item>

          <Form.Item
            name="targetScope"
            label="发放范围"
            rules={[{ required: true, message: "请选择发放范围" }]}
          >
            <Radio.Group>
              <Radio value="all">全体用户</Radio>
              <Radio value="level">按等级</Radio>
              <Radio value="users">指定用户</Radio>
            </Radio.Group>
          </Form.Item>

          {watchTargetScope === "level" && (
            <Form.Item
              name="levelCodes"
              label="会员等级"
              rules={[
                {
                  validator: (_, value: string[]) => {
                    if (!value || value.length === 0) {
                      return Promise.reject(new Error("请选择会员等级"));
                    }
                    return Promise.resolve();
                  },
                },
              ]}
            >
              <Select
                mode="multiple"
                placeholder="请选择会员等级"
                options={LEVEL_OPTIONS}
              />
            </Form.Item>
          )}

          {watchTargetScope === "users" && (
            <Form.Item
              name="userIdsInput"
              label="用户ID"
              rules={[
                {
                  validator: (_, value: string) => {
                    if (!value || parseUserIds(value).length === 0) {
                      return Promise.reject(new Error("请输入用户ID"));
                    }
                    return Promise.resolve();
                  },
                },
              ]}
            >
              <Input.TextArea
                rows={3}
                placeholder="多个用户ID用英文逗号分隔，如 1001,1002,1003"
              />
            </Form.Item>
          )}
        </Form>
      </Modal>
    );
  }
);

DistributeDialog.displayName = "DistributeDialog";

export default DistributeDialog;
