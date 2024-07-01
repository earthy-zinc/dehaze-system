import DatasetAPI from "@/api/dataset";
import { Dataset } from "@/api/dataset/model";
import {
  Form,
  Input,
  Modal,
  InputNumber,
  Select,
  Switch,
  Space,
  message,
} from "antd";
import { forwardRef, useImperativeHandle, useState } from "react";

const { Option } = Select;

const EditModal = forwardRef((props, ref) => {
  // 定义编辑对话框是否显示
  const [editModalVisible, setEditModalVisible] = useState(false);
  const [title, setTitle] = useState("");
  const [form] = Form.useForm();
  // 新增数据的父 id
  let parentId = -1;
  const [types, setTypes] = useState("");

  // 打开对话框
  const open = (operation: string, row: Dataset) => {
    // 设置对话框标题
    setTitle(`${operation}数据集基本信息`);
    setTypes(operation);
    if (operation === "编辑") {
      // 获取数据集单位 MB | DB
      const size = row.size;
      // 回填表单数据
      form.setFieldsValue({
        ...row,
        sizeNumber: +size.substring(0, size.length - 2),
        select: size.slice(-2),
      });
      console.log();
    } else {
      // 设置新增的数据集的父 id
      parentId = row.id;
    }
    setEditModalVisible(true);
  };

  // 子组件抛出 open 方法
  useImperativeHandle(ref, () => ({
    open,
  }));

  // 重置表单
  const onReset = () => {
    form.resetFields();
  };

  // 关闭对话框
  const onCancel = () => {
    setEditModalVisible(false);
    onReset();
  };

  type DataForm = Dataset & {
    sizeNumber: number;
    select: string;
  };

  // 提交表单
  const onFinish = (values: DataForm) => {
    // 拼接数据集大小
    const size = "" + values.sizeNumber + values.select;

    const formData: Dataset = {
      id: form.getFieldValue("id"),
      parentId,
      type: values.type,
      name: values.name,
      description: values.description,
      path: values.path,
      size,
      total: values.total,
      children: form.getFieldValue("children"),
      status: values.status ? 1 : 0,
    };

    if (types === "新增") {
      // 暂时不做校验
      DatasetAPI.add(formData).then((res) => {
        onCancel();
        message.success(`新增数据集成功`);
      });
    } else {
      DatasetAPI.update(formData.id, formData).then((res) => {
        onCancel();
        message.success(`修改数据集成功`);
      });
    }
  };

  return (
    <div>
      <Modal
        width={700}
        title={title}
        open={editModalVisible}
        okText="保存"
        cancelText="取消"
        okButtonProps={{ autoFocus: true, htmlType: "submit" }}
        onCancel={onCancel}
        destroyOnClose
        modalRender={(dom) => (
          <Form
            layout="horizontal"
            form={form}
            colon={false}
            name="编辑数据集信息"
            validateTrigger="onBlur"
            initialValues={{ total: 1, sizeNumber: 1, select: "MB" }}
            onFinish={onFinish}
          >
            {dom}
          </Form>
        )}
      >
        <Form.Item name="name" label="数据集名称" rules={[{ required: true }]}>
          <Input></Input>
        </Form.Item>
        <Form.Item name="type" label="数据集类别" rules={[{ required: true }]}>
          <Input></Input>
        </Form.Item>
        <Form.Item
          name="description"
          label="数据集描述"
          rules={[{ required: true }]}
        >
          <Input></Input>
        </Form.Item>
        <Form.Item label="&nbsp;&nbsp;&nbsp;数据集大小">
          <Space.Compact>
            <Form.Item
              name={["sizeNumber"]}
              rules={[{ required: true, message: "数据集大小不能为0" }]}
            >
              <InputNumber style={{ width: 120 }} min={0.01} />
            </Form.Item>
            <Form.Item name={["select"]}>
              <Select placeholder="Select">
                <Option value="MB">MB</Option>
                <Option value="DB">DB</Option>
              </Select>
            </Form.Item>
          </Space.Compact>
        </Form.Item>

        <Form.Item name="total" label="总图片数量" rules={[{ required: true }]}>
          <InputNumber style={{ width: 120 }} min={1} />
        </Form.Item>
        <Form.Item name="path" label="存 储 位 置" rules={[{ required: true }]}>
          <Input></Input>
        </Form.Item>
        <Form.Item name="status" label="&nbsp;&nbsp;&nbsp;数据集状态">
          <Switch
            checkedChildren="显示"
            unCheckedChildren="隐藏"
            defaultChecked
          />
        </Form.Item>
      </Modal>
    </div>
  );
});

EditModal.displayName = "EditModal";

export default EditModal;
