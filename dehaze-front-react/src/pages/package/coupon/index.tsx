import {
  CouponAPI,
  type CouponQuery,
  type CouponType,
  type CouponVO,
} from "dehaze-sdk-js";
import { useHasPerm } from "@/hooks/usePermission";
import {
  Button,
  Card,
  Form,
  Input,
  Popconfirm,
  Select,
  Space,
  Table,
  Tag,
  message,
  type TableColumnsType,
} from "antd";
import {
  DeleteOutlined,
  EditOutlined,
  GiftOutlined,
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
import CouponFormDialog, {
  type CouponFormDialogRef,
} from "./components/CouponFormDialog";
import DistributeDialog, {
  type DistributeDialogRef,
} from "./components/DistributeDialog";
import "./index.scss";

const TYPE_OPTIONS: { label: string; value: CouponType }[] = [
  { label: "满减券", value: "full_reduction" },
  { label: "折扣券", value: "discount" },
  { label: "无门槛券", value: "no_threshold" },
  { label: "体验券", value: "trial" },
];

const STATUS_OPTIONS = [
  { label: "启用", value: 1 },
  { label: "禁用", value: 0 },
];

const TYPE_TAG_COLOR: Record<string, string> = {
  full_reduction: "orange",
  discount: "green",
  no_threshold: "blue",
  trial: "red",
};

const TYPE_LABEL: Record<string, string> = {
  full_reduction: "满减券",
  discount: "折扣券",
  no_threshold: "无门槛券",
  trial: "体验券",
};

const formatFaceValue = (coupon: CouponVO) => {
  if (coupon.type === "discount") {
    return `${coupon.faceValue}折`;
  }
  return `¥${(coupon.faceValue ?? 0).toFixed(2)}`;
};

const formatValidRange = (coupon: CouponVO) => {
  if (coupon.validType === "fixed") {
    return `${coupon.validStart || "-"} ~ ${coupon.validEnd || "-"}`;
  }
  return `领取后 ${coupon.validDays ?? 0} 天`;
};

const CouponManagement: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [pageData, setPageData] = useState<CouponVO[]>([]);
  const [total, setTotal] = useState(0);
  const [searchForm] = Form.useForm();
  const [queryParams, setQueryParams] = useState<CouponQuery>({
    pageNum: 1,
    pageSize: 10,
  });
  const [refreshFlag, setRefreshFlag] = useState(0);

  const formDialogRef = useRef<CouponFormDialogRef>(null);
  const distributeDialogRef = useRef<DistributeDialogRef>(null);

  const hasPerm = useHasPerm();

  const loadData = useCallback(async (params: CouponQuery) => {
    setLoading(true);
    try {
      const result = await CouponAPI.getPage(params);
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
  }, []);

  const handleSearch = useCallback(
    (values: { name?: string; type?: CouponType; status?: number }) => {
      setQueryParams((prev) => ({
        ...prev,
        pageNum: 1,
        name: values.name || undefined,
        type: values.type,
        status: values.status,
      }));
    },
    []
  );

  const handleReset = useCallback(() => {
    searchForm.resetFields();
    setQueryParams({ pageNum: 1, pageSize: 10 });
  }, [searchForm]);

  const handlePageChange = useCallback((page: number, pageSize: number) => {
    setQueryParams((prev) => ({ ...prev, pageNum: page, pageSize }));
  }, []);

  const handleAdd = useCallback(() => {
    formDialogRef.current?.open("add");
  }, []);

  const handleEdit = useCallback((record: CouponVO) => {
    formDialogRef.current?.open("edit", record.id);
  }, []);

  const handleDelete = useCallback(
    (record: CouponVO) => {
      CouponAPI.deleteByIds(String(record.id))
        .then(() => {
          message.success(`优惠券「${record.name}」删除成功`);
          refreshList();
        })
        .catch((error) => {
          message.error(error?.message || "删除失败");
        });
    },
    [refreshList]
  );

  const handleDistribute = useCallback((record: CouponVO) => {
    distributeDialogRef.current?.open(record);
  }, []);

  const columns: TableColumnsType<CouponVO> = useMemo(
    () => [
      {
        title: "名称",
        dataIndex: "name",
        key: "name",
        width: 180,
        align: "center",
      },
      {
        title: "类型",
        dataIndex: "type",
        key: "type",
        width: 100,
        align: "center",
        render: (type: string) => (
          <Tag color={TYPE_TAG_COLOR[type] || "default"}>
            {TYPE_LABEL[type] || type}
          </Tag>
        ),
      },
      {
        title: "面值",
        dataIndex: "faceValue",
        key: "faceValue",
        width: 100,
        align: "right",
        render: (_: unknown, record: CouponVO) => (
          <span className="face-value">{formatFaceValue(record)}</span>
        ),
      },
      {
        title: "门槛",
        dataIndex: "threshold",
        key: "threshold",
        width: 100,
        align: "right",
        render: (threshold?: number) =>
          threshold ? (
            `满¥${(threshold ?? 0).toFixed(2)}`
          ) : (
            <span className="text-secondary">无门槛</span>
          ),
      },
      {
        title: "有效期",
        key: "validRange",
        width: 240,
        align: "center",
        render: (_: unknown, record: CouponVO) => formatValidRange(record),
      },
      {
        title: "总量/已领/已用",
        key: "qty",
        width: 140,
        align: "center",
        render: (_: unknown, record: CouponVO) =>
          `${record.totalQty} / ${record.issuedQty} / ${record.usedQty}`,
      },
      {
        title: "每人限领",
        dataIndex: "perUserLimit",
        key: "perUserLimit",
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
            {status === 1 ? "启用" : "禁用"}
          </Tag>
        ),
      },
      {
        title: "操作",
        key: "action",
        width: 220,
        align: "center",
        fixed: "right",
        render: (_: unknown, record: CouponVO) => (
          <Space size="small">
            {hasPerm("package:coupon:edit") && (
              <Button
                type="link"
                size="small"
                icon={<EditOutlined />}
                onClick={() => handleEdit(record)}
              >
                编辑
              </Button>
            )}
            {hasPerm("package:coupon:distribute") && (
              <Button
                type="link"
                size="small"
                icon={<GiftOutlined />}
                onClick={() => handleDistribute(record)}
              >
                发放
              </Button>
            )}
            {hasPerm("package:coupon:delete") && (
              <Popconfirm
                title={`确认删除优惠券「${record.name}」吗？删除后不可恢复。`}
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
    [handleEdit, handleDistribute, handleDelete, hasPerm]
  );

  return (
    <div className="coupon-management-container">
      <Card className="search-card" size="small">
        <Form form={searchForm} layout="inline" onFinish={handleSearch}>
          <Form.Item name="name" label="名称">
            <Input placeholder="优惠券名称" allowClear style={{ width: 180 }} />
          </Form.Item>
          <Form.Item name="type" label="类型">
            <Select
              placeholder="全部"
              allowClear
              style={{ width: 140 }}
              options={TYPE_OPTIONS}
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
          {hasPerm("package:coupon:add") && (
            <Button type="primary" icon={<PlusOutlined />} onClick={handleAdd}>
              新增
            </Button>
          )}
        </div>
        <Table
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

      <CouponFormDialog ref={formDialogRef} onSuccess={refreshList} />
      <DistributeDialog ref={distributeDialogRef} onSuccess={refreshList} />
    </div>
  );
};

export default CouponManagement;
