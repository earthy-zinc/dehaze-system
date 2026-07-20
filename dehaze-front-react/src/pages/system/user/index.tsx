import {
  DeptAPI,
  UserAPI,
  type DeptVO,
  type UserPageVO,
  type UserQuery,
} from "dehaze-sdk-js";
import { useDebounceFn } from "ahooks";
import {
  Button,
  Card,
  DatePicker,
  Dropdown,
  Empty,
  Form,
  Input,
  message,
  Modal,
  Popconfirm,
  Select,
  Space,
  Spin,
  Switch,
  Table,
  Tree,
  TreeSelect,
  Upload,
  type MenuProps,
  type TableColumnsType,
  type UploadFile,
} from "antd";
import {
  DeleteOutlined,
  DownOutlined,
  DownloadOutlined,
  EditOutlined,
  ExportOutlined,
  ImportOutlined,
  KeyOutlined,
  PlusOutlined,
  ReloadOutlined,
  SearchOutlined,
  UploadOutlined,
} from "@ant-design/icons";
import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import PasswordResetDialog, {
  type PasswordResetDialogRef,
} from "./components/PasswordResetDialog";
import UserFormDialog, {
  type UserFormDialogRef,
} from "./components/UserFormDialog";
import "./index.scss";

const { RangePicker } = DatePicker;

/** 递归转换部门数据为 Tree 组件需要的格式 */
function buildDeptTree(depts: DeptVO[]): any[] {
  return depts.map((dept) => ({
    title: dept.name,
    key: dept.id,
    children: dept.children?.length ? buildDeptTree(dept.children) : undefined,
  }));
}

/** 递归转换部门数据为 TreeSelect 组件需要的格式 */
function buildDeptTreeSelectData(depts: DeptVO[]): any[] {
  return depts.map((dept) => ({
    title: dept.name,
    value: dept.id,
    children: dept.children?.length
      ? buildDeptTreeSelectData(dept.children)
      : undefined,
  }));
}

/** 下载二进制数据为文件 */
function downloadBlob(data: ArrayBuffer, filename: string) {
  const blob = new Blob([data]);
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

const UserManagement: React.FC = () => {
  // ==================== 状态 ====================
  const [loading, setLoading] = useState(false);
  const [deptLoading, setDeptLoading] = useState(false);

  // 部门树
  const [deptList, setDeptList] = useState<DeptVO[]>([]);
  const [selectedDeptId, setSelectedDeptId] = useState<number | undefined>(
    undefined
  );

  // 查询参数
  const [queryParams, setQueryParams] = useState<UserQuery>({
    pageNum: 1,
    pageSize: 10,
  });

  // 表格数据
  const [userList, setUserList] = useState<UserPageVO[]>([]);
  const [total, setTotal] = useState(0);

  // 搜索表单
  const [searchForm] = Form.useForm();

  // 弹窗 ref
  const formDialogRef = useRef<UserFormDialogRef>(null);
  const passwordDialogRef = useRef<PasswordResetDialogRef>(null);

  // 选中行
  const [selectedRowKeys, setSelectedRowKeys] = useState<React.Key[]>([]);

  // 刷新标记
  const [refreshFlag, setRefreshFlag] = useState(0);

  // 状态切换中的用户ID（用于Switch loading）
  const [statusUpdatingId, setStatusUpdatingId] = useState<number | undefined>(
    undefined
  );

  // 导出loading
  const [exportLoading, setExportLoading] = useState(false);

  // 导入弹窗
  const [importDialogVisible, setImportDialogVisible] = useState(false);
  const [importLoading, setImportLoading] = useState(false);
  const [importDeptId, setImportDeptId] = useState<number | undefined>(
    undefined
  );
  const [importFileList, setImportFileList] = useState<UploadFile[]>([]);

  // ==================== 数据加载 ====================

  /** 加载部门树 */
  const loadDeptTree = useCallback(async () => {
    setDeptLoading(true);
    try {
      const data = await DeptAPI.getList();
      setDeptList(data || []);
      // 默认选中第一个根部门
      if (data && data.length > 0) {
        const rootId = data[0].id;
        setSelectedDeptId(rootId);
        setQueryParams((prev) => ({
          ...prev,
          pageNum: 1,
          deptId: rootId,
        }));
      }
    } finally {
      setDeptLoading(false);
    }
  }, []);

  /** 加载用户列表 */
  const loadUserList = useCallback(async (params: UserQuery) => {
    setLoading(true);
    try {
      const result = await UserAPI.getPage(params);
      setUserList(result.list || []);
      setTotal(result.total || 0);
    } finally {
      setLoading(false);
    }
  }, []);

  // ==================== 副作用 ====================

  useEffect(() => {
    loadDeptTree();
  }, [loadDeptTree]);

  useEffect(() => {
    loadUserList(queryParams);
  }, [queryParams, refreshFlag]);

  // ==================== 通用刷新 ====================

  const refreshList = useCallback(() => {
    setRefreshFlag((prev) => prev + 1);
    setSelectedRowKeys([]);
  }, []);

  // ==================== 事件处理 ====================

  /** 点击部门树节点 */
  const handleDeptSelect = useCallback((selectedKeys: React.Key[]) => {
    const deptId =
      selectedKeys.length > 0 ? (selectedKeys[0] as number) : undefined;
    setSelectedDeptId(deptId);
    setSelectedRowKeys([]);
    setQueryParams((prev) => ({
      ...prev,
      pageNum: 1,
      deptId,
    }));
  }, []);

  /** 搜索 */
  const handleSearch = useCallback(
    (values: {
      keywords?: string;
      status?: number;
      dateRange?: [any, any];
    }) => {
      const startTime =
        values.dateRange?.[0]?.format?.("YYYY-MM-DD HH:mm:ss") ?? undefined;
      const endTime =
        values.dateRange?.[1]?.format?.("YYYY-MM-DD HH:mm:ss") ?? undefined;
      setSelectedRowKeys([]);
      setQueryParams((prev) => ({
        ...prev,
        pageNum: 1,
        keywords: values.keywords || undefined,
        status: values.status !== undefined ? values.status : undefined,
        startTime,
        endTime,
      }));
    },
    []
  );

  /** 搜索防抖：关键字输入时延时触发搜索 */
  const { run: debouncedSearch } = useDebounceFn(
    () => {
      const values = searchForm.getFieldsValue();
      handleSearch(values);
    },
    { wait: 300 }
  );

  /** 重置搜索 */
  const handleReset = useCallback(() => {
    searchForm.resetFields();
    setSelectedRowKeys([]);
    setQueryParams((prev) => ({
      pageNum: 1,
      pageSize: prev.pageSize,
      deptId: selectedDeptId,
    }));
  }, [searchForm, selectedDeptId]);

  /** 分页变化 */
  const handlePageChange = useCallback((page: number, pageSize: number) => {
    setSelectedRowKeys([]);
    setQueryParams((prev) => ({
      ...prev,
      pageNum: page,
      pageSize,
    }));
  }, []);

  /** 新增 */
  const handleAdd = useCallback(() => {
    formDialogRef.current?.open("add");
  }, []);

  /** 编辑 */
  const handleEdit = useCallback((record: UserPageVO) => {
    formDialogRef.current?.open("edit", { id: record.id });
  }, []);

  /** 单个删除 */
  const handleDelete = useCallback(
    (record: UserPageVO) => {
      UserAPI.deleteByIds(String(record.id))
        .then(() => {
          message.success(`用户「${record.username}」删除成功`);
          refreshList();
        })
        .catch((error) => {
          message.error(error?.message || "删除失败");
        });
    },
    [refreshList]
  );

  /** 批量删除 */
  const handleBatchDelete = useCallback(() => {
    Modal.confirm({
      title: "批量删除",
      content: `确认删除选中的 ${selectedRowKeys.length} 个用户吗？删除后不可恢复。`,
      okText: "确定",
      cancelText: "取消",
      okType: "danger",
      onOk: () => {
        return UserAPI.deleteByIds(selectedRowKeys.join(","))
          .then(() => {
            message.success(`成功删除 ${selectedRowKeys.length} 个用户`);
            refreshList();
          })
          .catch((error) => {
            message.error(error?.message || "删除失败");
            return Promise.reject(error);
          });
      },
    });
  }, [selectedRowKeys, refreshList]);

  /** 重置密码 */
  const handleResetPassword = useCallback((record: UserPageVO) => {
    passwordDialogRef.current?.open(record.id!, record.username!);
  }, []);

  /** 切换用户状态 */
  const handleStatusChange = useCallback(
    (record: UserPageVO, checked: boolean) => {
      const newStatus = checked ? 1 : 0;
      setStatusUpdatingId(record.id);
      UserAPI.updateStatus(record.id!, newStatus)
        .then(() => {
          message.success(
            `用户「${record.username}」已${checked ? "启用" : "禁用"}`
          );
          refreshList();
        })
        .catch((error) => {
          message.error(error?.message || "状态切换失败");
        })
        .finally(() => setStatusUpdatingId(undefined));
    },
    [refreshList]
  );

  /** 下载导入模板 */
  const handleDownloadTemplate = useCallback(async () => {
    try {
      const data = await UserAPI.downloadTemplate();
      downloadBlob(data, "用户导入模板.xlsx");
      message.success("模板下载成功");
    } catch (error: any) {
      message.error(error?.message || "模板下载失败");
    }
  }, []);

  /** 导出用户列表 */
  const handleExport = useCallback(async () => {
    setExportLoading(true);
    try {
      const data = await UserAPI.export(queryParams);
      downloadBlob(data, "用户列表.xlsx");
      message.success("导出成功");
    } catch (error: any) {
      message.error(error?.message || "导出失败");
    } finally {
      setExportLoading(false);
    }
  }, [queryParams]);

  /** 打开导入弹窗 */
  const handleOpenImport = useCallback(() => {
    setImportDeptId(selectedDeptId);
    setImportFileList([]);
    setImportDialogVisible(true);
  }, [selectedDeptId]);

  /** 提交导入 */
  const handleImportSubmit = useCallback(async () => {
    if (importDeptId === undefined) {
      message.warning("请选择所属部门");
      return;
    }
    const file = importFileList[0]?.originFileObj as File | undefined;
    if (!file) {
      message.warning("请选择Excel文件");
      return;
    }
    setImportLoading(true);
    try {
      await UserAPI.import(importDeptId, file);
      message.success("导入成功");
      setImportDialogVisible(false);
      refreshList();
    } catch (error: any) {
      message.error(error?.message || "导入失败");
    } finally {
      setImportLoading(false);
    }
  }, [importDeptId, importFileList, refreshList]);

  /** 导入下拉菜单点击 */
  const importMenuItems: MenuProps["items"] = useMemo(
    () => [
      { key: "template", label: "下载模板", icon: <DownloadOutlined /> },
      { key: "import", label: "导入数据", icon: <ImportOutlined /> },
    ],
    []
  );

  const handleImportMenuClick = useCallback(
    ({ key }: { key: string }) => {
      if (key === "template") {
        handleDownloadTemplate();
      } else if (key === "import") {
        handleOpenImport();
      }
    },
    [handleDownloadTemplate, handleOpenImport]
  );

  // ==================== 表格列定义 ====================

  const columns: TableColumnsType<UserPageVO> = useMemo(
    () => [
      {
        title: "编号",
        dataIndex: "id",
        key: "id",
        width: 100,
        align: "center",
      },
      {
        title: "用户名",
        dataIndex: "username",
        key: "username",
        width: 120,
        align: "center",
      },
      {
        title: "昵称",
        dataIndex: "nickname",
        key: "nickname",
        width: 120,
        align: "center",
      },
      {
        title: "性别",
        dataIndex: "genderLabel",
        key: "genderLabel",
        width: 80,
        align: "center",
      },
      {
        title: "部门",
        dataIndex: "deptName",
        key: "deptName",
        width: 120,
        align: "center",
      },
      {
        title: "手机号",
        dataIndex: "mobile",
        key: "mobile",
        width: 120,
        align: "center",
        render: (text: string) => text || "-",
      },
      {
        title: "状态",
        dataIndex: "status",
        key: "status",
        width: 80,
        align: "center",
        render: (status: number, record: UserPageVO) => (
          <Switch
            checked={status === 1}
            loading={statusUpdatingId === record.id}
            onChange={(checked) => handleStatusChange(record, checked)}
          />
        ),
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
        render: (_: unknown, record: UserPageVO) => (
          <Space size="small">
            <Button
              type="link"
              size="small"
              icon={<KeyOutlined />}
              onClick={() => handleResetPassword(record)}
            >
              重置密码
            </Button>
            <Button
              type="link"
              size="small"
              icon={<EditOutlined />}
              onClick={() => handleEdit(record)}
            >
              编辑
            </Button>
            <Popconfirm
              title={`确认删除用户「${record.username}」吗？删除后不可恢复。`}
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
    [
      handleEdit,
      handleDelete,
      handleResetPassword,
      handleStatusChange,
      statusUpdatingId,
    ]
  );

  /** 行选择配置 */
  const rowSelection = useMemo(
    () => ({
      selectedRowKeys,
      onChange: (keys: React.Key[]) => setSelectedRowKeys(keys),
    }),
    [selectedRowKeys]
  );

  // ==================== 渲染 ====================

  return (
    <div className="user-management-container">
      {/* 左侧部门树 */}
      <Card className="dept-tree-card" size="small" title="部门列表">
        {deptLoading ? (
          <div className="dept-tree-loading">
            <Spin />
          </div>
        ) : deptList.length === 0 ? (
          <Empty description="暂无部门数据" />
        ) : (
          <Tree
            treeData={buildDeptTree(deptList)}
            selectedKeys={selectedDeptId ? [selectedDeptId] : []}
            onSelect={handleDeptSelect}
            defaultExpandAll
            blockNode
          />
        )}
      </Card>

      {/* 右侧用户列表区域 */}
      <div className="user-list-area">
        {/* 搜索区域 */}
        <Card className="search-card" size="small">
          <Form
            form={searchForm}
            layout="inline"
            onFinish={handleSearch}
            onValuesChange={(changed) => {
              // 关键字输入时触发防抖搜索
              if ("keywords" in changed) {
                debouncedSearch();
              }
            }}
          >
            <Form.Item name="keywords" label="关键字">
              <Input
                placeholder="用户名/昵称/手机号"
                allowClear
                style={{ width: 220 }}
              />
            </Form.Item>
            <Form.Item name="status" label="状态">
              <Select
                placeholder="全部"
                allowClear
                style={{ width: 120 }}
                options={[
                  { value: 1, label: "启用" },
                  { value: 0, label: "禁用" },
                ]}
              />
            </Form.Item>
            <Form.Item name="dateRange" label="创建时间">
              <RangePicker
                showTime
                placeholder={["开始时间", "截止时间"]}
                style={{ width: 360 }}
              />
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
                <Button
                  danger
                  icon={<DeleteOutlined />}
                  disabled={selectedRowKeys.length === 0}
                  onClick={handleBatchDelete}
                >
                  删除
                </Button>
                {/* 导入下拉菜单 */}
                <Dropdown
                  menu={{
                    items: importMenuItems,
                    onClick: handleImportMenuClick,
                  }}
                >
                  <Button icon={<ImportOutlined />}>
                    导入 <DownOutlined />
                  </Button>
                </Dropdown>
                {/* 导出 */}
                <Button
                  icon={<ExportOutlined />}
                  loading={exportLoading}
                  onClick={handleExport}
                >
                  导出
                </Button>
              </Space>
            </Form.Item>
          </Form>
        </Card>

        {/* 表格区域 */}
        <Card className="table-card" size="small">
          <Table
            rowSelection={rowSelection}
            columns={columns}
            dataSource={userList}
            rowKey={(record) => record.id ?? Math.random()}
            loading={loading}
            scroll={{ x: 1280 }}
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
      </div>

      {/* 新增/编辑弹窗 */}
      <UserFormDialog ref={formDialogRef} onSuccess={refreshList} />

      {/* 密码重置弹窗 */}
      <PasswordResetDialog ref={passwordDialogRef} onSuccess={refreshList} />

      {/* 导入弹窗 */}
      <Modal
        title="导入用户"
        open={importDialogVisible}
        confirmLoading={importLoading}
        onOk={handleImportSubmit}
        onCancel={() => setImportDialogVisible(false)}
        okText="确定"
        cancelText="取消"
        destroyOnHidden
      >
        <Form layout="vertical">
          <Form.Item label="所属部门" required>
            <TreeSelect
              treeData={buildDeptTreeSelectData(deptList)}
              value={importDeptId}
              onChange={(value: number) => setImportDeptId(value)}
              placeholder="请选择部门"
              allowClear
              treeDefaultExpandAll
            />
          </Form.Item>
          <Form.Item label="Excel文件" required>
            <Upload
              accept=".xls,.xlsx"
              maxCount={1}
              fileList={importFileList}
              beforeUpload={(file) => {
                setImportFileList([
                  {
                    uid: file.uid,
                    name: file.name,
                    status: "done",
                    originFileObj: file,
                  } as UploadFile,
                ]);
                return false;
              }}
              onRemove={() => setImportFileList([])}
            >
              <Button icon={<UploadOutlined />}>选择文件</Button>
            </Upload>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default UserManagement;
