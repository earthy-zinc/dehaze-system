import { DeptAPI, type DeptQuery, type DeptVO } from "dehaze-sdk-js";
import ImportExportToolbar from "@/components/ImportExportToolbar";
import {
  Button,
  Card,
  Form,
  Input,
  message,
  Popconfirm,
  Select,
  Space,
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
import DeptFormDialog, {
  type DeptFormDialogRef,
} from "./components/DeptFormDialog";
import { useHasPerm } from "@/hooks/usePermission";
import "./index.scss";

/** 状态映射 */
const STATUS_MAP: Record<number, { label: string; color: string }> = {
  1: { label: "启用", color: "green" },
  0: { label: "禁用", color: "default" },
};

const DeptManagement: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [deptList, setDeptList] = useState<DeptVO[]>([]);
  const [searchForm] = Form.useForm();
  const [queryParams, setQueryParams] = useState<DeptQuery>({});

  const dialogRef = useRef<DeptFormDialogRef>(null);
  const [refreshFlag, setRefreshFlag] = useState(0);

  // 权限校验
  const hasPerm = useHasPerm();

  // ==================== 数据加载 ====================

  const loadDeptList = useCallback(async (params: DeptQuery) => {
    setLoading(true);
    try {
      const data = await DeptAPI.getList(params);
      setDeptList(data || []);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadDeptList(queryParams);
  }, [queryParams, refreshFlag]);

  const refreshList = useCallback(() => {
    setRefreshFlag((prev) => prev + 1);
  }, []);

  // ==================== 事件处理 ====================

  const handleSearch = useCallback(
    (values: { name?: string; status?: number }) => {
      setQueryParams({
        keywords: values.name || undefined,
        status: values.status !== undefined ? values.status : undefined,
      });
    },
    []
  );

  const handleReset = useCallback(() => {
    searchForm.resetFields();
    setQueryParams({});
  }, [searchForm]);

  const handleAdd = useCallback(() => {
    dialogRef.current?.open("add");
  }, []);

  const handleAddSub = useCallback((record: DeptVO) => {
    dialogRef.current?.open("addSub", record);
  }, []);

  const handleEdit = useCallback((record: DeptVO) => {
    dialogRef.current?.open("edit", record);
  }, []);

  const handleDelete = useCallback(
    (record: DeptVO) => {
      DeptAPI.deleteByIds(String(record.id))
        .then(() => {
          message.success(`部门「${record.name}」删除成功`);
          refreshList();
        })
        .catch((error) => {
          message.error(error?.message || "删除失败");
        });
    },
    [refreshList]
  );

  // ==================== 表格列定义 ====================

  const columns: TableColumnsType<DeptVO> = useMemo(
    () => [
      {
        title: "部门名称",
        dataIndex: "name",
        key: "name",
        align: "left" as const,
      },
      {
        title: "状态",
        dataIndex: "status",
        key: "status",
        width: 100,
        align: "center",
        render: (status: number) => {
          const info = STATUS_MAP[status] || {
            label: "未知",
            color: "default",
          };
          return <Tag color={info.color}>{info.label}</Tag>;
        },
      },
      {
        title: "排序",
        dataIndex: "sort",
        key: "sort",
        width: 100,
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
        width: 200,
        align: "center",
        fixed: "right",
        render: (_: unknown, record: DeptVO) => (
          <Space size="small">
            {hasPerm("sys:dept:add") && (
              <Button
                type="link"
                size="small"
                icon={<PlusOutlined />}
                onClick={() => handleAddSub(record)}
              >
                新增下级
              </Button>
            )}
            {hasPerm("sys:dept:edit") && (
              <Button
                type="link"
                size="small"
                icon={<EditOutlined />}
                onClick={() => handleEdit(record)}
              >
                编辑
              </Button>
            )}
            {hasPerm("sys:dept:delete") && (
              <Popconfirm
                title={`确认删除部门「${record.name}」吗？删除后不可恢复。`}
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
    [handleAddSub, handleEdit, handleDelete, hasPerm]
  );

  // ==================== 渲染 ====================

  return (
    <div className="dept-management-container">
      {/* 搜索区域 */}
      <Card className="search-card" size="small">
        <Form form={searchForm} layout="inline" onFinish={handleSearch}>
          <Form.Item name="name" label="部门名称">
            <Input placeholder="部门名称" allowClear style={{ width: 200 }} />
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
              {hasPerm("sys:dept:add") && (
                <Button
                  type="primary"
                  icon={<PlusOutlined />}
                  onClick={handleAdd}
                >
                  新增
                </Button>
              )}
              <ImportExportToolbar
                module="dept"
                queryParams={queryParams}
                onImportComplete={refreshList}
              />
            </Space>
          </Form.Item>
        </Form>
      </Card>

      {/* 表格区域 */}
      <Card className="table-card" size="small">
        <Table
          columns={columns}
          dataSource={deptList}
          rowKey={(record) => record.id ?? Math.random()}
          loading={loading}
          expandable={{
            defaultExpandAllRows: true,
            indentSize: 30,
          }}
          pagination={false}
          scroll={{ x: 800 }}
        />
      </Card>

      <DeptFormDialog ref={dialogRef} onSuccess={refreshList} />
    </div>
  );
};

export default DeptManagement;
