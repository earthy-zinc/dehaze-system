import {
  DeptAPI,
  DictAPI,
  RoleAPI,
  UserAPI,
  type DeptVO,
  type OptionType,
  type UserForm,
} from "dehaze-sdk-js";
import {
  Form,
  Input,
  Modal,
  Radio,
  Select,
  TreeSelect,
  message,
} from "antd";
import React, {
  forwardRef,
  useCallback,
  useImperativeHandle,
  useState,
} from "react";

/** 性别字典类型编码 */
const GENDER_DICT_CODE = "gender";

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

export interface UserFormDialogRef {
  open: (type: "add" | "edit", record?: UserForm) => void;
}

interface UserFormDialogProps {
  onSuccess?: () => void;
}

const UserFormDialog = forwardRef<UserFormDialogRef, UserFormDialogProps>(
  ({ onSuccess }, ref) => {
  const [visible, setVisible] = useState(false);
  const [dialogType, setDialogType] = useState<"add" | "edit">("add");
  const [confirmLoading, setConfirmLoading] = useState(false);
  const [form] = Form.useForm<UserForm>();

  // 下拉数据
  const [deptTree, setDeptTree] = useState<any[]>([]);
  const [roleOptions, setRoleOptions] = useState<OptionType[]>([]);
  const [genderOptions, setGenderOptions] = useState<OptionType[]>([]);

  /** 加载下拉选项数据 */
  const loadOptions = useCallback(async () => {
    const [deptData, roleData, genderData] = await Promise.all([
      DeptAPI.getList().catch(() => [] as DeptVO[]),
      RoleAPI.getOptions().catch(() => [] as OptionType[]),
      DictAPI.getDictOptions(GENDER_DICT_CODE).catch(() => [] as OptionType[]),
    ]);
    setDeptTree(buildDeptTreeSelect(deptData || []));
    setRoleOptions(roleData || []);
    setGenderOptions(genderData || []);
  }, []);

  /** 打开弹窗 */
  const open = useCallback(
    async (type: "add" | "edit", record?: UserForm) => {
      setDialogType(type);
      setVisible(true);

      // 并行加载下拉选项
      loadOptions();

      if (type === "add") {
        form.resetFields();
        form.setFieldsValue({ status: 1 });
      } else if (record?.id) {
        form.resetFields();
        try {
          const data = await UserAPI.getFormData(record.id);
          form.setFieldsValue({
            id: data.id ?? record.id,
            username: data.username,
            nickname: data.nickname,
            deptId: data.deptId,
            gender: data.gender,
            roleIds: data.roleIds,
            mobile: data.mobile,
            email: data.email,
            status: data.status ?? 1,
          });
        } catch {
          message.error("获取用户信息失败");
        }
      }
    },
    [form, loadOptions]
  );

  useImperativeHandle(ref, () => ({ open }), [open]);

  /** 关闭弹窗 */
  const handleCancel = useCallback(() => {
    setVisible(false);
    form.resetFields();
  }, [form]);

  /** 提交表单 */
  const handleSubmit = useCallback(async () => {
    try {
      const values = await form.validateFields();
      setConfirmLoading(true);

      if (dialogType === "add") {
        await UserAPI.add(values);
        message.success("新增用户成功");
      } else {
        const userId = form.getFieldValue("id");
        await UserAPI.update(userId, values);
        message.success("修改用户成功");
      }

      handleCancel();
      onSuccess?.();
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } catch (error: any) {
      if (error?.errorFields) return; // 表单校验失败，不做处理
      message.error(error?.message || "操作失败");
    } finally {
      setConfirmLoading(false);
    }
  }, [form, dialogType, handleCancel, onSuccess]);

  return (
    <Modal
      title={dialogType === "add" ? "新增用户" : "修改用户"}
      open={visible}
      width={800}
      confirmLoading={confirmLoading}
      okText="保存"
      cancelText="取消"
      destroyOnClose
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
          name="username"
          label="用户名"
          rules={[
            { required: true, message: "请输入用户名" },
            { max: 30, message: "用户名不能超过30个字符" },
          ]}
        >
          <Input
            placeholder="请输入用户名"
            disabled={dialogType === "edit"}
          />
        </Form.Item>

        <Form.Item
          name="nickname"
          label="用户昵称"
          rules={[
            { required: true, message: "请输入用户昵称" },
            { max: 30, message: "昵称不能超过30个字符" },
          ]}
        >
          <Input placeholder="请输入用户昵称" />
        </Form.Item>

        <Form.Item
          name="deptId"
          label="所属部门"
          rules={[{ required: true, message: "请选择所属部门" }]}
        >
          <TreeSelect
            treeData={deptTree}
            placeholder="请选择所属部门"
            treeDefaultExpandAll
            allowClear
          />
        </Form.Item>

        <Form.Item
          name="gender"
          label="性别"
          rules={[{ required: true, message: "请选择性别" }]}
        >
          <Select
            placeholder="请选择性别"
            allowClear
            options={genderOptions.map((opt) => ({
              value: Number(opt.value),
              label: opt.label,
            }))}
          />
        </Form.Item>

        <Form.Item
          name="roleIds"
          label="角色"
          rules={[{ required: true, message: "请选择角色" }]}
        >
          <Select
            mode="multiple"
            placeholder="请选择角色"
            allowClear
            options={roleOptions.map((opt) => ({
              value: Number(opt.value),
              label: opt.label,
            }))}
          />
        </Form.Item>

        <Form.Item
          name="mobile"
          label="手机号"
          rules={[
            {
              pattern: /^1[3-9]\d{9}$/,
              message: "请输入正确的手机号",
            },
          ]}
        >
          <Input placeholder="请输入手机号" />
        </Form.Item>

        <Form.Item
          name="email"
          label="邮箱"
          rules={[
            {
              type: "email",
              message: "请输入正确的邮箱格式",
            },
          ]}
        >
          <Input placeholder="请输入邮箱" />
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
});

UserFormDialog.displayName = "UserFormDialog";

export default UserFormDialog;
