import {
  OrderAPI,
  type RefundAuditForm,
  type RefundRecordVO,
} from "dehaze-sdk-js";
import { Form, Input, Modal, Tag, message } from "antd";
import React, {
  forwardRef,
  useCallback,
  useImperativeHandle,
  useState,
} from "react";

export interface RefundAuditDialogRef {
  open: (record: RefundRecordVO, approved: boolean) => void;
}

interface RefundAuditDialogProps {
  onSuccess?: () => void;
}

const RefundAuditDialog = forwardRef<
  RefundAuditDialogRef,
  RefundAuditDialogProps
>(({ onSuccess }, ref) => {
  const [visible, setVisible] = useState(false);
  const [confirmLoading, setConfirmLoading] = useState(false);
  const [approved, setApproved] = useState(true);
  const [record, setRecord] = useState<RefundRecordVO | null>(null);
  const [form] = Form.useForm<RefundAuditForm>();

  const open = useCallback(
    (row: RefundRecordVO, isApproved: boolean) => {
      setRecord(row);
      setApproved(isApproved);
      form.resetFields();
      form.setFieldsValue({ approved: isApproved, remark: "" });
      setVisible(true);
    },
    [form]
  );

  useImperativeHandle(ref, () => ({ open }), [open]);

  const handleCancel = useCallback(() => {
    setVisible(false);
    setRecord(null);
    form.resetFields();
  }, [form]);

  const handleSubmit = useCallback(async () => {
    if (!record) return;
    try {
      const values = await form.validateFields();
      setConfirmLoading(true);
      const payload: RefundAuditForm = {
        approved,
        remark: values.remark,
      };
      const action = approved
        ? OrderAPI.approveRefund(record.id, payload)
        : OrderAPI.rejectRefund(record.id, payload);
      await action;
      message.success(approved ? "审核通过" : "已驳回");
      handleCancel();
      onSuccess?.();
    } catch (error: any) {
      if (error?.errorFields) return;
      message.error(error?.message || "操作失败");
    } finally {
      setConfirmLoading(false);
    }
  }, [record, approved, form, handleCancel, onSuccess]);

  return (
    <Modal
      title={approved ? "退款审核通过" : "退款审核驳回"}
      open={visible}
      width={500}
      confirmLoading={confirmLoading}
      okText="确定"
      cancelText="取消"
      destroyOnHidden
      onOk={handleSubmit}
      onCancel={handleCancel}
    >
      {record && (
        <Form
          form={form}
          layout="horizontal"
          labelCol={{ span: 6 }}
          wrapperCol={{ span: 16 }}
          colon={false}
        >
          <Form.Item label="退款单号">
            <span>{record.refundNo}</span>
          </Form.Item>
          <Form.Item label="订单号">
            <span>{record.orderNo}</span>
          </Form.Item>
          <Form.Item label="用户">
            <span>{record.username}</span>
          </Form.Item>
          <Form.Item label="退款金额">
            <span>¥{record.refundAmount.toFixed(2)}</span>
          </Form.Item>
          <Form.Item label="审核结果">
            <Tag color={approved ? "success" : "error"}>
              {approved ? "通过" : "驳回"}
            </Tag>
          </Form.Item>
          <Form.Item
            name="remark"
            label="审核备注"
            rules={[{ required: true, message: "请输入审核备注" }]}
          >
            <Input.TextArea rows={3} placeholder="请输入审核备注" />
          </Form.Item>
        </Form>
      )}
    </Modal>
  );
});

RefundAuditDialog.displayName = "RefundAuditDialog";

export default RefundAuditDialog;
