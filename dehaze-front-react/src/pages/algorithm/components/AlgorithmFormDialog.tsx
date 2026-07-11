import { AlgorithmAPI, type Algorithm, type CreateAlgorithmOptional } from "dehaze-sdk-js";
import { Form, Input, Modal, Switch, TreeSelect, message } from "antd";
import React, { forwardRef, useCallback, useImperativeHandle, useState } from "react";

/** 递归转换算法树为 TreeSelect 格式 */
function buildTreeSelect(algorithms: Algorithm[]): any[] {
  return algorithms.map((a) => ({
    title: a.name,
    value: a.id,
    children: a.children?.length ? buildTreeSelect(a.children) : undefined,
  }));
}

export interface AlgorithmFormDialogRef {
  open: (type: "add" | "edit" | "addSub", record?: Algorithm) => void;
}

interface Props {
  onSuccess?: () => void;
}

const AlgorithmFormDialog = forwardRef<AlgorithmFormDialogRef, Props>(
  ({ onSuccess }, ref) => {
    const [visible, setVisible] = useState(false);
    const [dialogType, setDialogType] = useState<"add" | "edit" | "addSub">("add");
    const [confirmLoading, setConfirmLoading] = useState(false);
    const [form] = Form.useForm();
    const [treeData, setTreeData] = useState<any[]>([]);

    const loadTree = useCallback(async () => {
      try {
        const data = await AlgorithmAPI.getList();
        setTreeData([
          { title: "顶级算法", value: 0 },
          ...buildTreeSelect(Array.isArray(data) ? data : []),
        ]);
      } catch {
        setTreeData([{ title: "顶级算法", value: 0 }]);
      }
    }, []);

    const open = useCallback(
      async (type: "add" | "edit" | "addSub", record?: Algorithm) => {
        setDialogType(type);
        setVisible(true);
        loadTree();

        if (type === "add") {
          form.resetFields();
          form.setFieldsValue({ parentId: 0, status: 1 });
        } else if (type === "addSub" && record) {
          form.resetFields();
          form.setFieldsValue({ parentId: record.id, status: 1 });
        } else if (type === "edit" && record) {
          form.resetFields();
          form.setFieldsValue({
            id: record.id,
            parentId: record.parentId ?? 0,
            name: record.name,
            type: record.type,
            description: record.description,
            importPath: record.importPath,
            flops: record.flops,
            params: record.params,
            status: record.status ?? 1,
          });
        }
      },
      [form, loadTree]
    );

    useImperativeHandle(ref, () => ({ open }), [open]);

    const handleCancel = useCallback(() => { setVisible(false); form.resetFields(); }, [form]);

    const handleSubmit = useCallback(async () => {
      try {
        const values = await form.validateFields();
        setConfirmLoading(true);

        const formData: CreateAlgorithmOptional = {
          parentId: values.parentId ?? 0,
          name: values.name,
          type: values.type,
          description: values.description || "",
          importPath: values.importPath,
          flops: values.flops,
          params: values.params,
          status: values.status ?? 1,
        };

        if (dialogType === "edit") {
          const id = form.getFieldValue("id");
          await AlgorithmAPI.update(id, formData as any);
          message.success("修改算法成功");
        } else {
          await AlgorithmAPI.add(formData as any);
          message.success("新增算法成功");
        }

        handleCancel();
        onSuccess?.();
      } catch (error: any) {
        if (error?.errorFields) return;
        message.error(error?.message || "操作失败");
      } finally { setConfirmLoading(false); }
    }, [form, dialogType, handleCancel, onSuccess]);

    const title = dialogType === "edit" ? "修改算法" : "新增算法";

    return (
      <Modal
        title={title} open={visible} width={700} confirmLoading={confirmLoading}
        okText="保存" cancelText="取消" destroyOnClose onOk={handleSubmit} onCancel={handleCancel}
      >
        <Form form={form} layout="vertical" colon={false} validateTrigger="onBlur">
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 24px" }}>
            <Form.Item name="parentId" label="上级算法">
              <TreeSelect treeData={treeData} placeholder="请选择上级算法" treeDefaultExpandAll allowClear />
            </Form.Item>
            <Form.Item name="type" label="算法类型" rules={[{ required: true, message: "请输入算法类型" }]}>
              <Input placeholder="如 deep_learning / traditional" disabled={dialogType === "edit"} />
            </Form.Item>
          </div>
          <Form.Item name="name" label="算法名称" rules={[{ required: true, message: "请输入算法名称" }]}>
            <Input placeholder="请输入算法名称" />
          </Form.Item>
          <Form.Item name="description" label="算法描述">
            <Input.TextArea placeholder="请输入算法描述" rows={2} />
          </Form.Item>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 24px" }}>
            <Form.Item name="importPath" label="代码导入路径">
              <Input placeholder="如 models.ridcp" />
            </Form.Item>
            <Form.Item name="status" label="状态" valuePropName="checked" getValueFromEvent={(checked: boolean) => checked ? 1 : 0}>
              <Switch checkedChildren="启用" unCheckedChildren="禁用" />
            </Form.Item>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 24px" }}>
            <Form.Item name="flops" label="FLOPs">
              <Input placeholder="如 1.5G" />
            </Form.Item>
            <Form.Item name="params" label="参数量">
              <Input placeholder="如 1.2M" />
            </Form.Item>
          </div>
        </Form>
      </Modal>
    );
  }
);

AlgorithmFormDialog.displayName = "AlgorithmFormDialog";
export default AlgorithmFormDialog;
