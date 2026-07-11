import { MenuAPI, type MenuQuery, type MenuVO } from "dehaze-sdk-js";
import { type DisPatchType, RootState } from "@/store";
import { generateRoutes } from "@/store/modules/permissionSlice";
import {
  Button,
  Card,
  Form,
  Input,
  message,
  Popconfirm,
  Space,
  Switch,
  Table,
  Tag,
  type TableColumnsType,
} from "antd";
import {
  DeleteOutlined,
  EditOutlined,
  PlusOutlined,
  ReloadOutlined,
  SearchOutlined,
} from "@ant-design/icons";
import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useDispatch, useSelector } from "react-redux";
import MenuFormDialog, {
  type MenuFormDialogRef,
} from "./components/MenuFormDialog";
import "./index.scss";

/** 菜单类型映射 */
const TYPE_MAP: Record<string, { label: string; color: string }> = {
  CATALOG: { label: "目录", color: "blue" },
  MENU: { label: "菜单", color: "green" },
  BUTTON: { label: "按钮", color: "orange" },
  EXTLINK: { label: "外链", color: "purple" },
};

const MenuManagement: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [menuList, setMenuList] = useState<MenuVO[]>([]);
  const [searchForm] = Form.useForm();
  const [queryParams, setQueryParams] = useState<MenuQuery>({});

  const dialogRef = useRef<MenuFormDialogRef>(null);
  const [refreshFlag, setRefreshFlag] = useState(0);

  const dispatch = useDispatch<DisPatchType>();
  const roles = useSelector((state: RootState) => state.user.user.roles);

  const loadData = useCallback(async (params: MenuQuery) => {
    setLoading(true);
    try {
      const data = await MenuAPI.getList(params);
      setMenuList(data || []);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData(queryParams);
  }, [queryParams, refreshFlag]);

  const refreshList = useCallback(() => setRefreshFlag((prev) => prev + 1), []);

  /** 刷新列表并重新生成路由 */
  const refreshListAndRoutes = useCallback(() => {
    refreshList();
    dispatch(generateRoutes(roles));
  }, [refreshList, dispatch, roles]);

  const handleSearch = useCallback((values: { keywords?: string }) => {
    setQueryParams({ keywords: values.keywords || undefined });
  }, []);

  const handleReset = useCallback(() => {
    searchForm.resetFields();
    setQueryParams({});
  }, [searchForm]);

  const handleAdd = useCallback(() => dialogRef.current?.open("add"), []);
  const handleEdit = useCallback(
    (record: MenuVO) => dialogRef.current?.open("edit", record),
    []
  );

  /** 切换菜单显示状态 */
  const handleToggleVisible = useCallback(
    async (record: MenuVO) => {
      try {
        // 获取完整表单数据后修改 visible 字段
        const formData = await MenuAPI.getFormData(record.id!);
        formData.visible = record.visible === 1 ? 0 : 1;
        await MenuAPI.update(String(record.id), formData);
        message.success(
          `菜单「${record.name}」${formData.visible === 1 ? "显示" : "隐藏"}成功`
        );
        refreshListAndRoutes();
      } catch (error: any) {
        message.error(error?.message || "操作失败");
      }
    },
    [refreshListAndRoutes]
  );

  const handleDelete = useCallback(
    (record: MenuVO) => {
      MenuAPI.deleteById(record.id!)
        .then(() => {
          message.success(`菜单「${record.name}」删除成功`);
          refreshListAndRoutes();
        })
        .catch((error) => message.error(error?.message || "删除失败"));
    },
    [refreshListAndRoutes]
  );

  const columns: TableColumnsType<MenuVO> = useMemo(
    () => [
      {
        title: "菜单名称",
        dataIndex: "name",
        key: "name",
        width: 200,
        align: "left" as const,
      },
      {
        title: "图标",
        dataIndex: "icon",
        key: "icon",
        width: 80,
        align: "center",
        render: (text: string) => text || "-",
      },
      {
        title: "类型",
        dataIndex: "type",
        key: "type",
        width: 100,
        align: "center",
        render: (type: string) => {
          const info = TYPE_MAP[type] || {
            label: type || "未知",
            color: "default",
          };
          return <Tag color={info.color}>{info.label}</Tag>;
        },
      },
      {
        title: "路由地址",
        dataIndex: "path",
        key: "path",
        width: 150,
        render: (text: string) => text || "-",
      },
      {
        title: "权限标识",
        dataIndex: "perm",
        key: "perm",
        width: 150,
        render: (text: string) => text || "-",
      },
      {
        title: "显示状态",
        dataIndex: "visible",
        key: "visible",
        width: 100,
        align: "center",
        render: (_: unknown, record: MenuVO) => (
          <Popconfirm
            title={`确认${record.visible === 1 ? "隐藏" : "显示"}该菜单吗？`}
            onConfirm={() => handleToggleVisible(record)}
            okText="确定"
            cancelText="取消"
          >
            <Switch checked={record.visible === 1} />
          </Popconfirm>
        ),
      },
      {
        title: "排序",
        dataIndex: "sort",
        key: "sort",
        width: 80,
        align: "center",
      },
      {
        title: "操作",
        key: "action",
        width: 160,
        align: "center",
        fixed: "right",
        render: (_: unknown, record: MenuVO) => (
          <Space size="small">
            <Button
              type="link"
              size="small"
              icon={<EditOutlined />}
              onClick={() => handleEdit(record)}
            >
              编辑
            </Button>
            <Popconfirm
              title={`确认删除菜单「${record.name}」吗？删除后不可恢复。`}
              onConfirm={() => handleDelete(record)}
              okText="确定"
              cancelText="取消"
              okType="danger"
            >
              <Button type="link" size="small" danger icon={<DeleteOutlined />}>
                删除
              </Button>
            </Popconfirm>
          </Space>
        ),
      },
    ],
    [handleEdit, handleDelete, handleToggleVisible]
  );

  return (
    <div className="menu-management-container">
      <Card className="search-card" size="small">
        <Form form={searchForm} layout="inline" onFinish={handleSearch}>
          <Form.Item name="keywords" label="菜单名称">
            <Input placeholder="菜单名称" allowClear style={{ width: 200 }} />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button
                type="primary"
                htmlType="submit"
                icon={<SearchOutlined />}
              >
                搜索
              </Button>
              <Button
                htmlType="reset"
                icon={<ReloadOutlined />}
                onClick={handleReset}
              >
                重置
              </Button>
              <Button
                type="primary"
                icon={<PlusOutlined />}
                onClick={handleAdd}
              >
                新增
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Card>

      <Card className="table-card" size="small">
        <Table
          columns={columns}
          dataSource={menuList}
          rowKey={(record) => record.id ?? Math.random()}
          loading={loading}
          expandable={{ defaultExpandAllRows: true, indentSize: 30 }}
          pagination={false}
          scroll={{ x: 1100 }}
        />
      </Card>

      <MenuFormDialog ref={dialogRef} onSuccess={refreshListAndRoutes} />
    </div>
  );
};

export default MenuManagement;
