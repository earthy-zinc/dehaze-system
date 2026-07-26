import { RoleAPI, type RolePageVO, type RoleQuery } from "dehaze-sdk-js";
import ImportExportToolbar from "@/components/ImportExportToolbar";
import { useHasPerm } from "@/hooks/usePermission";
import {
  Button,
  Card,
  Form,
  Input,
  message,
  Modal,
  Popconfirm,
  Space,
  Switch,
  Table,
  type TableColumnsType,
} from "antd";
import {
  DeleteOutlined,
  EditOutlined,
  PlusOutlined,
  ReloadOutlined,
  SafetyOutlined,
  SearchOutlined,
} from "@ant-design/icons";
import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import PermissionDialog, {
  type PermissionDialogRef,
} from "./components/PermissionDialog";
import RoleFormDialog, {
  type RoleFormDialogRef,
} from "./components/RoleFormDialog";
import "./index.scss";

/** 数据权限映射 */
const DATA_SCOPE_MAP: Record<number, string> = {
  0: "全部数据",
  1: "部门及子部门数据",
  2: "本部门数据",
  3: "本人数据",
};

/** 状态映射 */
const STATUS_MAP: Record<number, { label: string; color: string }> = {
  1: { label: "启用", color: "green" },
  0: { label: "禁用", color: "default" },
};

/** 获取数据权限标签文本 */
function getDataScopeLabel(
  record: RolePageVO & { dataScope?: number; dataScopeLabel?: string }
): string {
  if (record.dataScopeLabel) return record.dataScopeLabel;
  if (record.dataScope !== undefined)
    return DATA_SCOPE_MAP[record.dataScope] || "未知";
  return "-";
}

const RoleManagement: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [pageData, setPageData] = useState<RolePageVO[]>([]);
  const [total, setTotal] = useState(0);
  const [searchForm] = Form.useForm();
  const [queryParams, setQueryParams] = useState<RoleQuery>({
    pageNum: 1,
    pageSize: 10,
  });
  const [selectedRowKeys, setSelectedRowKeys] = useState<React.Key[]>([]);

  const formDialogRef = useRef<RoleFormDialogRef>(null);
  const permissionDialogRef = useRef<PermissionDialogRef>(null);
  const [refreshFlag, setRefreshFlag] = useState(0);

  // 权限校验
  const hasPerm = useHasPerm();

  // ==================== 数据加载 ====================

  const loadData = useCallback(async (params: RoleQuery) => {
    setLoading(true);
    try {
      const result = await RoleAPI.getPage(params);
      setPageData(result.list || []);
      setTotal(result.total || 0);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData(queryParams);
  }, [queryParams, refreshFlag]);

  const refreshList = useCallback(() => {
    setRefreshFlag((prev) => prev + 1);
    setSelectedRowKeys([]);
  }, []);

  // ==================== 事件处理 ====================

  const handleSearch = useCallback((values: { keywords?: string }) => {
    setSelectedRowKeys([]);
    setQueryParams((prev) => ({
      ...prev,
      pageNum: 1,
      keywords: values.keywords || undefined,
    }));
  }, []);

  const handleReset = useCallback(() => {
    searchForm.resetFields();
    setSelectedRowKeys([]);
    setQueryParams({ pageNum: 1, pageSize: 10 });
  }, [searchForm]);

  const handlePageChange = useCallback((page: number, pageSize: number) => {
    setSelectedRowKeys([]);
    setQueryParams((prev) => ({ ...prev, pageNum: page, pageSize }));
  }, []);

  const handleAdd = useCallback(() => {
    formDialogRef.current?.open("add");
  }, []);

  const handleEdit = useCallback((record: RolePageVO) => {
    formDialogRef.current?.open("edit", record.id);
  }, []);

  const handleDelete = useCallback(
    (record: RolePageVO) => {
      RoleAPI.deleteByIds(String(record.id))
        .then(() => {
          message.success(`角色「${record.name}」删除成功`);
          refreshList();
        })
        .catch((error) => {
          message.error(error?.message || "删除失败");
        });
    },
    [refreshList]
  );

  const handleBatchDelete = useCallback(() => {
    Modal.confirm({
      title: "批量删除",
      content: `确认删除选中的 ${selectedRowKeys.length} 个角色吗？删除后不可恢复。`,
      okText: "确定",
      cancelText: "取消",
      okType: "danger",
      onOk: () =>
        RoleAPI.deleteByIds(selectedRowKeys.join(","))
          .then(() => {
            message.success(`成功删除 ${selectedRowKeys.length} 个角色`);
            refreshList();
          })
          .catch((error) => {
            message.error(error?.message || "删除失败");
            return Promise.reject(error);
          }),
    });
  }, [selectedRowKeys, refreshList]);

  const handleStatusChange = useCallback(
    (record: RolePageVO, checked: boolean) => {
      const newStatus = checked ? 1 : 0;
      RoleAPI.updateStatus(record.id!, newStatus)
        .then(() => {
          message.success(
            `角色「${record.name}」已${checked ? "启用" : "禁用"}`
          );
          refreshList();
        })
        .catch((error) => {
          message.error(error?.message || "状态切换失败");
        });
    },
    [refreshList]
  );

  const handleAssignPermission = useCallback((record: RolePageVO) => {
    permissionDialogRef.current?.open(record.id!, record.name!);
  }, []);

  // ==================== 表格列定义 ====================

  const columns: TableColumnsType<RolePageVO> = useMemo(
    () => [
      {
        title: "角色名称",
        dataIndex: "name",
        key: "name",
        width: 150,
        align: "center",
      },
      {
        title: "角色编码",
        dataIndex: "code",
        key: "code",
        width: 150,
        align: "center",
      },
      {
        title: "数据权限",
        key: "dataScope",
        width: 160,
        align: "center",
        render: (_: unknown, record: RolePageVO) =>
          getDataScopeLabel(record as any),
      },
      {
        title: "状态",
        dataIndex: "status",
        key: "status",
        width: 80,
        align: "center",
        render: (status: number, record: RolePageVO) => (
          <Switch
            checked={status === 1}
            onChange={(checked) => handleStatusChange(record, checked)}
          />
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
        title: "创建时间",
        dataIndex: "createTime",
        key: "createTime",
        width: 180,
        align: "center",
      },
      {
        title: "操作",
        key: "action",
        width: 240,
        align: "center",
        fixed: "right",
        render: (_: unknown, record: RolePageVO) => (
          <Space size="small">
            {hasPerm("sys:role:edit") && (
              <Button
                type="link"
                size="small"
                icon={<SafetyOutlined />}
                onClick={() => handleAssignPermission(record)}
              >
                分配权限
              </Button>
            )}
            {hasPerm("sys:role:edit") && (
              <Button
                type="link"
                size="small"
                icon={<EditOutlined />}
                onClick={() => handleEdit(record)}
              >
                编辑
              </Button>
            )}
            {hasPerm("sys:role:delete") && (
              <Popconfirm
                title={`确认删除角色「${record.name}」吗？删除后不可恢复。`}
                onConfirm={() => handleDelete(record)}
                okText="确定"
                cancelText="取消"
                okType="danger"
              >
                <Button
                  type="link"
                  size="small"
                  danger
                  icon={<DeleteOutlined />}
                >
                  删除
                </Button>
              </Popconfirm>
            )}
          </Space>
        ),
      },
    ],
    [
      handleStatusChange,
      handleAssignPermission,
      handleEdit,
      handleDelete,
      hasPerm,
    ]
  );

  const rowSelection = useMemo(
    () => ({
      selectedRowKeys,
      onChange: (keys: React.Key[]) => setSelectedRowKeys(keys),
    }),
    [selectedRowKeys]
  );

  // ==================== 渲染 ====================

  return (
    <div className="role-management-container">
      <Card className="search-card" size="small">
        <Form form={searchForm} layout="inline" onFinish={handleSearch}>
          <Form.Item name="keywords" label="角色名称">
            <Input placeholder="角色名称" allowClear style={{ width: 200 }} />
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
              {hasPerm("sys:role:add") && (
                <Button
                  type="primary"
                  icon={<PlusOutlined />}
                  onClick={handleAdd}
                >
                  新增
                </Button>
              )}
              {hasPerm("sys:role:delete") && (
                <Button
                  danger
                  icon={<DeleteOutlined />}
                  disabled={selectedRowKeys.length === 0}
                  onClick={handleBatchDelete}
                >
                  删除
                </Button>
              )}
              <ImportExportToolbar
                module="role"
                queryParams={queryParams}
                onImportComplete={refreshList}
              />
            </Space>
          </Form.Item>
        </Form>
      </Card>

      <Card className="table-card" size="small">
        <Table
          rowSelection={rowSelection}
          columns={columns}
          dataSource={pageData}
          rowKey={(record) => record.id ?? Math.random()}
          loading={loading}
          scroll={{ x: 1100 }}
          pagination={{
            current: queryParams.pageNum,
            pageSize: queryParams.pageSize,
            total,
            showSizeChanger: true,
            showQuickJumper: true,
            pageSizeOptions: ["10", "20", "50", "100"],
            showTotal: (t) => `共 ${t} 条`,
            onChange: handlePageChange,
          }}
        />
      </Card>

      <RoleFormDialog ref={formDialogRef} onSuccess={refreshList} />
      <PermissionDialog ref={permissionDialogRef} onSuccess={refreshList} />
    </div>
  );
};

export default RoleManagement;
