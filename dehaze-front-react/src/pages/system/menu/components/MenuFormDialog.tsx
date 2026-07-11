import { MenuAPI, type MenuForm, type MenuVO } from "dehaze-sdk-js";
import {
  ApartmentOutlined,
  AppstoreOutlined,
  BookOutlined,
  BugOutlined,
  BuildOutlined,
  BulbOutlined,
  CalendarOutlined,
  ClockCircleOutlined,
  CloudOutlined,
  CodeOutlined,
  DashboardOutlined,
  DatabaseOutlined,
  DeleteOutlined,
  DesktopOutlined,
  EditOutlined,
  EnvironmentOutlined,
  EyeOutlined,
  FileOutlined,
  FolderOutlined,
  GlobalOutlined,
  HomeOutlined,
  LockOutlined,
  MailOutlined,
  MenuOutlined,
  PlusOutlined,
  ReloadOutlined,
  RocketOutlined,
  SafetyOutlined,
  SearchOutlined,
  SettingOutlined,
  ShoppingOutlined,
  SolutionOutlined,
  StarOutlined,
  TableOutlined,
  TeamOutlined,
  ThunderboltOutlined,
  ToolOutlined,
  UserOutlined,
} from "@ant-design/icons";
import {
  Form,
  Input,
  InputNumber,
  Modal,
  Radio,
  Select,
  Space,
  TreeSelect,
  message,
} from "antd";
import React, {
  forwardRef,
  useCallback,
  useImperativeHandle,
  useState,
} from "react";

/** 菜单类型选项 */
const TYPE_OPTIONS = [
  { value: "CATALOG", label: "目录" },
  { value: "MENU", label: "菜单" },
  { value: "BUTTON", label: "按钮" },
  { value: "EXTLINK", label: "外链" },
];

/** 常用图标映射（图标名称 -> 图标组件） */
const ICON_MAP: Record<string, React.ReactNode> = {
  HomeOutlined: <HomeOutlined />,
  SettingOutlined: <SettingOutlined />,
  MenuOutlined: <MenuOutlined />,
  ApartmentOutlined: <ApartmentOutlined />,
  BookOutlined: <BookOutlined />,
  SafetyOutlined: <SafetyOutlined />,
  UserOutlined: <UserOutlined />,
  DatabaseOutlined: <DatabaseOutlined />,
  AppstoreOutlined: <AppstoreOutlined />,
  DashboardOutlined: <DashboardOutlined />,
  FileOutlined: <FileOutlined />,
  FolderOutlined: <FolderOutlined />,
  ToolOutlined: <ToolOutlined />,
  CloudOutlined: <CloudOutlined />,
  CodeOutlined: <CodeOutlined />,
  EyeOutlined: <EyeOutlined />,
  EditOutlined: <EditOutlined />,
  DeleteOutlined: <DeleteOutlined />,
  PlusOutlined: <PlusOutlined />,
  SearchOutlined: <SearchOutlined />,
  ReloadOutlined: <ReloadOutlined />,
  StarOutlined: <StarOutlined />,
  MailOutlined: <MailOutlined />,
  LockOutlined: <LockOutlined />,
  TeamOutlined: <TeamOutlined />,
  SolutionOutlined: <SolutionOutlined />,
  TableOutlined: <TableOutlined />,
  CalendarOutlined: <CalendarOutlined />,
  ClockCircleOutlined: <ClockCircleOutlined />,
  EnvironmentOutlined: <EnvironmentOutlined />,
  GlobalOutlined: <GlobalOutlined />,
  BugOutlined: <BugOutlined />,
  DesktopOutlined: <DesktopOutlined />,
  ThunderboltOutlined: <ThunderboltOutlined />,
  BulbOutlined: <BulbOutlined />,
  RocketOutlined: <RocketOutlined />,
  BuildOutlined: <BuildOutlined />,
  ShoppingOutlined: <ShoppingOutlined />,
};

/** 图标下拉选项 */
const ICON_OPTIONS = Object.keys(ICON_MAP).map((name) => ({
  value: name,
  label: name,
}));

/** 递归转换菜单数据为 TreeSelect 格式，过滤掉按钮和外链类型 */
function buildMenuTreeSelect(menus: MenuVO[]): any[] {
  return menus
    .filter((menu) => menu.type === "CATALOG" || menu.type === "MENU")
    .map((menu) => ({
      title: menu.name,
      value: menu.id,
      children: menu.children?.length
        ? buildMenuTreeSelect(menu.children)
        : undefined,
    }));
}

export interface MenuFormDialogRef {
  open: (type: "add" | "edit", record?: MenuVO) => void;
}

interface MenuFormDialogProps {
  onSuccess?: () => void;
}

const MenuFormDialog = forwardRef<MenuFormDialogRef, MenuFormDialogProps>(
  ({ onSuccess }, ref) => {
    const [visible, setVisible] = useState(false);
    const [dialogType, setDialogType] = useState<"add" | "edit">("add");
    const [confirmLoading, setConfirmLoading] = useState(false);
    const [form] = Form.useForm();
    const [menuType, setMenuType] = useState<string>("CATALOG");

    const [menuTree, setMenuTree] = useState<any[]>([]);

    const loadMenuOptions = useCallback(async () => {
      try {
        // 使用 getList 获取完整菜单数据（含 type 字段）以支持类型过滤
        const data = await MenuAPI.getList({});
        setMenuTree([
          { title: "顶级菜单", value: 0 },
          ...buildMenuTreeSelect(data || []),
        ]);
      } catch {
        setMenuTree([{ title: "顶级菜单", value: 0 }]);
      }
    }, []);

    const open = useCallback(
      async (type: "add" | "edit", record?: MenuVO) => {
        setDialogType(type);
        setVisible(true);
        loadMenuOptions();

        if (type === "add") {
          form.resetFields();
          form.setFieldsValue({
            type: "CATALOG",
            visible: 1,
            sort: 1,
            parentId: 0,
          });
          setMenuType("CATALOG");
        } else if (type === "edit" && record?.id) {
          form.resetFields();
          try {
            const data = await MenuAPI.getFormData(record.id);
            form.setFieldsValue({
              id: data.id ?? String(record.id),
              name: data.name,
              type: data.type,
              parentId: data.parentId ?? 0,
              path: data.path,
              component: data.component,
              perm: data.perm,
              icon: data.icon,
              redirect: data.redirect,
              visible: data.visible,
              sort: data.sort,
              keepAlive: data.keepAlive,
              alwaysShow: data.alwaysShow,
            });
            setMenuType(data.type || "CATALOG");
          } catch {
            message.error("获取菜单信息失败");
          }
        }
      },
      [form, loadMenuOptions]
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
          await MenuAPI.update(id, values);
          message.success("修改菜单成功");
        } else {
          await MenuAPI.add(values);
          message.success("新增菜单成功");
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

    const handleTypeChange = useCallback(
      (value: string) => {
        setMenuType(value);
        // 切换类型时清除条件字段
        form.setFieldsValue({
          path: undefined,
          component: undefined,
          perm: undefined,
          icon: undefined,
          redirect: undefined,
        });
      },
      [form]
    );

    // 条件显示控制
    const showPath = menuType === "MENU" || menuType === "EXTLINK";
    const showComponent = menuType === "MENU";
    const showPerm = menuType === "BUTTON";
    const showIcon = menuType === "CATALOG" || menuType === "MENU";
    const showRedirect = menuType === "CATALOG";

    // 条件校验规则
    const pathRequired = menuType === "MENU" || menuType === "EXTLINK";
    const componentRequired = menuType === "MENU";
    const permRequired = menuType === "BUTTON";

    return (
      <Modal
        title={dialogType === "add" ? "新增菜单" : "修改菜单"}
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
          layout="vertical"
          colon={false}
          validateTrigger="onBlur"
        >
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: "0 24px",
            }}
          >
            <Form.Item
              name="type"
              label="菜单类型"
              rules={[{ required: true, message: "请选择菜单类型" }]}
            >
              <Select options={TYPE_OPTIONS} onChange={handleTypeChange} />
            </Form.Item>

            <Form.Item name="parentId" label="上级菜单">
              <TreeSelect
                treeData={menuTree}
                placeholder="请选择上级菜单"
                treeDefaultExpandAll
                allowClear
              />
            </Form.Item>
          </div>

          <Form.Item
            name="name"
            label="菜单名称"
            rules={[
              { required: true, message: "请输入菜单名称" },
              { min: 2, max: 64, message: "菜单名称长度应在 2-64 字符之间" },
            ]}
          >
            <Input placeholder="请输入菜单名称" />
          </Form.Item>

          {showPath && (
            <Form.Item
              name="path"
              label={menuType === "EXTLINK" ? "外链地址" : "路由地址"}
              rules={
                pathRequired
                  ? menuType === "MENU"
                    ? [
                        { required: true, message: "路由地址不能为空" },
                        { pattern: /^\//, message: "路由地址必须以 / 开头" },
                      ]
                    : [
                        { required: true, message: "外链地址不能为空" },
                        {
                          pattern: /^https?:\/\/.+/,
                          message: "外链地址必须以 http:// 或 https:// 开头",
                        },
                      ]
                  : undefined
              }
            >
              <Input
                placeholder={
                  menuType === "EXTLINK"
                    ? "请输入外链地址（如 https://）"
                    : "请输入路由地址（如 /system/user）"
                }
              />
            </Form.Item>
          )}

          {showComponent && (
            <Form.Item
              name="component"
              label="组件路径"
              rules={
                componentRequired
                  ? [{ required: true, message: "组件路径不能为空" }]
                  : undefined
              }
            >
              <Input placeholder="请输入组件路径（如 system/user/index）" />
            </Form.Item>
          )}

          {showPerm && (
            <Form.Item
              name="perm"
              label="权限标识"
              rules={
                permRequired
                  ? [
                      { required: true, message: "权限标识不能为空" },
                      {
                        pattern: /^[a-z]+:[a-z]+:[a-z]+$/,
                        message: "格式：模块:功能:操作（如 sys:menu:add）",
                      },
                    ]
                  : undefined
              }
            >
              <Input placeholder="请输入权限标识（如 sys:menu:add）" />
            </Form.Item>
          )}

          {showIcon && (
            <Form.Item name="icon" label="图标">
              <Select
                placeholder="请选择图标"
                allowClear
                showSearch
                options={ICON_OPTIONS}
                optionRender={(option) => (
                  <Space>
                    {ICON_MAP[option.value as string]}
                    <span>{option.label}</span>
                  </Space>
                )}
              />
            </Form.Item>
          )}

          {showRedirect && (
            <Form.Item name="redirect" label="路由重定向">
              <Input placeholder="请输入重定向路径" />
            </Form.Item>
          )}

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: "0 24px",
            }}
          >
            <Form.Item name="visible" label="显示状态">
              <Radio.Group>
                <Radio value={1}>显示</Radio>
                <Radio value={0}>隐藏</Radio>
              </Radio.Group>
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
          </div>
        </Form>
      </Modal>
    );
  }
);

MenuFormDialog.displayName = "MenuFormDialog";

export default MenuFormDialog;
