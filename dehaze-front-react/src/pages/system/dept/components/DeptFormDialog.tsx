import { DeptAPI, type DeptForm, type DeptVO } from "dehaze-sdk-js";
import {
  Form,
  Input,
  InputNumber,
  Modal,
  Radio,
  TreeSelect,
  message,
} from "antd";
import React, {
  forwardRef,
  useCallback,
  useImperativeHandle,
  useState,
} from "react";

/** 递归转换部门数据为 TreeSelect 需要的格式 */
function buildDeptTreeSelect(depts: DeptVO[]): any[] {
  return depts.map((dept) => ({
    title: dept.name,
    value: dept.id,
    children: dept.children?.length
      ? buildDeptTreeSelect(dept.children)
      : undefined,
  }));
}

export interface DeptFormDialogRef {
  open: (type: "add" | "edit" | "addSub", record?: DeptVO) => void;
}

interface DeptFormDialogProps {
  onSuccess?: () => void;
}

const DeptFormDialog = forwardRef<DeptFormDialogRef, DeptFormDialogProps>(
  ({ onSuccess }, ref) => {
    const [visible, setVisible] = useState(false);
    const [dialogType, setDialogType] = useState<"add" | "edit" | "addSub">(
      "add"
    );
    const [confirmLoading, setConfirmLoading] = useState(false);
    const [form] = Form.useForm<DeptForm>();

    const [deptTree, setDeptTree] = useState<any[]>([]);

    /** 加载部门树选项 */
    const loadDeptOptions = useCallback(async () => {
      try {
        const data = await DeptAPI.getList();
        setDeptTree(buildDeptTreeSelect(data || []));
      } catch {
        setDeptTree([]);
      }
    }, []);

    const open = useCallback(
      async (type: "add" | "edit" | "addSub", record?: DeptVO) => {
        setDialogType(type);
        setVisible(true);
        loadDeptOptions();

        if (type === "add") {
          form.resetFields();
          form.setFieldsValue({ sort: 1, status: 1 });
        } else if (type === "addSub" && record) {
          form.resetFields();
          form.setFieldsValue({
            parentId: record.id,
            sort: 1,
            status: 1,
          });
        } else if (type === "edit" && record?.id) {
          form.resetFields();
          try {
            const data = await DeptAPI.getFormData(record.id);
            form.setFieldsValue({
              id: data.id ?? record.id,
              name: data.name,
              parentId: data.parentId,
              sort: data.sort ?? 1,
              status: data.status ?? 1,
            });
          } catch {
            message.error("获取部门信息失败");
          }
        }
      },
      [form, loadDeptOptions]
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
          const deptId = form.getFieldValue("id");
          await DeptAPI.update(deptId, values);
          message.success("修改部门成功");
        } else {
          await DeptAPI.add(values);
          message.success("新增部门成功");
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

    const dialogTitle = dialogType === "edit" ? "修改部门" : "新增部门";

    return (
      <Modal
        title={dialogTitle}
        open={visible}
        width={600}
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
          <Form.Item
            name="name"
            label="部门名称"
            rules={[
              { required: true, message: "请输入部门名称" },
              { max: 30, message: "部门名称不能超过30个字符" },
            ]}
          >
            <Input placeholder="请输入部门名称" />
          </Form.Item>

          <Form.Item
            name="parentId"
            label="上级部门"
            rules={[{ required: true, message: "请选择上级部门" }]}
          >
            <TreeSelect
              treeData={deptTree}
              placeholder="请选择上级部门"
              treeDefaultExpandAll
              allowClear
            />
          </Form.Item>

          <Form.Item
            name="sort"
            label="排序"
            rules={[{ required: true, message: "请输入排序值" }]}
          >
            <InputNumber
              min={1}
              style={{ width: "100%" }}
              placeholder="请输入排序值"
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

DeptFormDialog.displayName = "DeptFormDialog";

export default DeptFormDialog;
