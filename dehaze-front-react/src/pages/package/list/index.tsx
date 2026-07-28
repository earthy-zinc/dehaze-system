import {
  PackageAPI,
  type PackageLevelCode,
  type PackagePageVO,
  type PackagePeriod,
  type PackageQuery,
  type PackageStatus,
} from "dehaze-sdk-js";
import { useHasPerm } from "@/hooks/usePermission";
import {
  Button,
  Card,
  DatePicker,
  Form,
  Input,
  Modal,
  Popconfirm,
  Select,
  Space,
  Table,
  Tag,
  message,
  type TableColumnsType,
} from "antd";
import {
  BarChartOutlined,
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
import dayjs from "dayjs";
import PackageFormDialog, {
  type PackageFormDialogRef,
} from "./components/PackageFormDialog";
import SalesStatsDrawer, {
  type SalesStatsDrawerRef,
} from "./components/SalesStatsDrawer";
import "./index.scss";

const { RangePicker } = DatePicker;

const LEVEL_OPTIONS = [
  { value: "level_1", label: "基础版" },
  { value: "level_2", label: "专业版" },
  { value: "level_3", label: "旗舰版" },
];

const PERIOD_OPTIONS = [
  { value: "monthly", label: "月卡" },
  { value: "quarterly", label: "季卡" },
  { value: "yearly", label: "年卡" },
];

const STATUS_OPTIONS = [
  { value: 1, label: "在售" },
  { value: 0, label: "下架" },
];

const LEVEL_TAG_COLOR: Record<string, string> = {
  level_1: "blue",
  level_2: "purple",
  level_3: "gold",
};

const PERIOD_LABEL: Record<string, string> = {
  monthly: "月卡",
  quarterly: "季卡",
  yearly: "年卡",
};

const PackageManagement: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [pageData, setPageData] = useState<PackagePageVO[]>([]);
  const [total, setTotal] = useState(0);
  const [searchForm] = Form.useForm();
  const [queryParams, setQueryParams] = useState<PackageQuery>({
    pageNum: 1,
    pageSize: 10,
  });
  const [selectedRowKeys, setSelectedRowKeys] = useState<React.Key[]>([]);
  const [refreshFlag, setRefreshFlag] = useState(0);

  const formDialogRef = useRef<PackageFormDialogRef>(null);
  const statsDrawerRef = useRef<SalesStatsDrawerRef>(null);

  const hasPerm = useHasPerm();

  const loadData = useCallback(async (params: PackageQuery) => {
    setLoading(true);
    try {
      const result = await PackageAPI.getPage(params);
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

  const handleSearch = useCallback(
    (values: {
      name?: string;
      levelCode?: PackageLevelCode;
      period?: PackagePeriod;
      status?: PackageStatus;
      createTimeRange?: [dayjs.Dayjs, dayjs.Dayjs];
    }) => {
      setSelectedRowKeys([]);
      setQueryParams((prev) => ({
        ...prev,
        pageNum: 1,
        name: values.name || undefined,
        levelCode: values.levelCode,
        period: values.period,
        status: values.status,
        startTime: values.createTimeRange?.[0]?.format("YYYY-MM-DD"),
        endTime: values.createTimeRange?.[1]?.format("YYYY-MM-DD"),
      }));
    },
    []
  );

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

  const handleEdit = useCallback((record: PackagePageVO) => {
    formDialogRef.current?.open("edit", record.id);
  }, []);

  const handleDelete = useCallback(
    (record: PackagePageVO) => {
      PackageAPI.deleteByIds(String(record.id))
        .then(() => {
          message.success(`套餐「${record.name}」删除成功`);
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
      content: `确认删除选中的 ${selectedRowKeys.length} 个套餐吗？删除后不可恢复。`,
      okText: "确定",
      cancelText: "取消",
      okType: "danger",
      onOk: () =>
        PackageAPI.deleteByIds(selectedRowKeys.join(","))
          .then(() => {
            message.success(`成功删除 ${selectedRowKeys.length} 个套餐`);
            refreshList();
          })
          .catch((error) => {
            message.error(error?.message || "删除失败");
            return Promise.reject(error);
          }),
    });
  }, [selectedRowKeys, refreshList]);

  const handleToggleStatus = useCallback(
    (record: PackagePageVO) => {
      const next: PackageStatus = record.status === 1 ? 0 : 1;
      const text = next === 1 ? "上架" : "下架";
      PackageAPI.updateStatus(record.id, next)
        .then(() => {
          message.success(`套餐「${record.name}」${text}成功`);
          refreshList();
        })
        .catch((error) => {
          message.error(error?.message || `${text}失败`);
        });
    },
    [refreshList]
  );

  const handleOpenStats = useCallback(() => {
    statsDrawerRef.current?.open();
  }, []);

  const columns: TableColumnsType<PackagePageVO> = useMemo(
    () => [
      {
        title: "套餐名",
        dataIndex: "name",
        key: "name",
        width: 160,
        align: "center",
      },
      {
        title: "等级",
        dataIndex: "levelCode",
        key: "levelCode",
        width: 110,
        align: "center",
        render: (levelCode: string, record) => (
          <Tag color={LEVEL_TAG_COLOR[levelCode] || "default"}>
            {record.levelName}
          </Tag>
        ),
      },
      {
        title: "计费周期",
        dataIndex: "period",
        key: "period",
        width: 100,
        align: "center",
        render: (period: string) => PERIOD_LABEL[period] || period,
      },
      {
        title: "原价",
        dataIndex: "originalPrice",
        key: "originalPrice",
        width: 100,
        align: "right",
        render: (v: number) => `¥${(v ?? 0).toFixed(2)}`,
      },
      {
        title: "售价",
        dataIndex: "salePrice",
        key: "salePrice",
        width: 100,
        align: "right",
        render: (v: number) => (
          <span className="sale-price">¥{(v ?? 0).toFixed(2)}</span>
        ),
      },
      {
        title: "日均",
        dataIndex: "dailyPrice",
        key: "dailyPrice",
        width: 110,
        align: "right",
        render: (v: number) => `¥${(v ?? 0).toFixed(2)}/天`,
      },
      {
        title: "销量",
        dataIndex: "salesCount",
        key: "salesCount",
        width: 90,
        align: "center",
      },
      {
        title: "状态",
        dataIndex: "status",
        key: "status",
        width: 90,
        align: "center",
        render: (status: number) => (
          <Tag color={status === 1 ? "success" : "default"} bordered={false}>
            {status === 1 ? "在售" : "下架"}
          </Tag>
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
        width: 220,
        align: "center",
        fixed: "right",
        render: (_: unknown, record: PackagePageVO) => (
          <Space size="small">
            {hasPerm("package:edit") && (
              <Button
                type="link"
                size="small"
                icon={<EditOutlined />}
                onClick={() => handleEdit(record)}
              >
                编辑
              </Button>
            )}
            {hasPerm("package:edit") && (
              <Popconfirm
                title={`确认${
                  record.status === 1 ? "下架" : "上架"
                }套餐「${record.name}」吗？`}
                onConfirm={() => handleToggleStatus(record)}
                okText="确定"
                cancelText="取消"
              >
                <Button type="link" size="small" danger={record.status === 1}>
                  {record.status === 1 ? "下架" : "上架"}
                </Button>
              </Popconfirm>
            )}
            {hasPerm("package:delete") && (
              <Popconfirm
                title={`确认删除套餐「${record.name}」吗？删除后不可恢复。`}
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
    [handleToggleStatus, handleEdit, handleDelete, hasPerm]
  );

  const rowSelection = useMemo(
    () => ({
      selectedRowKeys,
      onChange: (keys: React.Key[]) => setSelectedRowKeys(keys),
    }),
    [selectedRowKeys]
  );

  return (
    <div className="package-management-container">
      <Card className="search-card" size="small">
        <Form form={searchForm} layout="inline" onFinish={handleSearch}>
          <Form.Item name="name" label="套餐名">
            <Input placeholder="套餐名" allowClear style={{ width: 160 }} />
          </Form.Item>
          <Form.Item name="levelCode" label="等级">
            <Select
              placeholder="全部"
              allowClear
              style={{ width: 140 }}
              options={LEVEL_OPTIONS}
            />
          </Form.Item>
          <Form.Item name="period" label="计费周期">
            <Select
              placeholder="全部"
              allowClear
              style={{ width: 140 }}
              options={PERIOD_OPTIONS}
            />
          </Form.Item>
          <Form.Item name="status" label="状态">
            <Select
              placeholder="全部"
              allowClear
              style={{ width: 120 }}
              options={STATUS_OPTIONS}
            />
          </Form.Item>
          <Form.Item name="createTimeRange" label="创建时间">
            <RangePicker style={{ width: 240 }} />
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
            </Space>
          </Form.Item>
        </Form>
      </Card>

      <Card className="table-card" size="small">
        <div className="table-toolbar">
          <Space>
            {hasPerm("package:add") && (
              <Button
                type="primary"
                icon={<PlusOutlined />}
                onClick={handleAdd}
              >
                新增
              </Button>
            )}
            {hasPerm("package:delete") && (
              <Button
                danger
                icon={<DeleteOutlined />}
                disabled={selectedRowKeys.length === 0}
                onClick={handleBatchDelete}
              >
                批量删除
              </Button>
            )}
          </Space>
          {hasPerm("package:sales") && (
            <Button icon={<BarChartOutlined />} onClick={handleOpenStats}>
              销售统计
            </Button>
          )}
        </div>
        <Table
          rowSelection={rowSelection}
          columns={columns}
          dataSource={pageData}
          rowKey={(record) => record.id ?? Math.random()}
          loading={loading}
          scroll={{ x: 1300 }}
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

      <PackageFormDialog ref={formDialogRef} onSuccess={refreshList} />
      <SalesStatsDrawer ref={statsDrawerRef} />
    </div>
  );
};

export default PackageManagement;
