import {
  OrderAPI,
  type OrderPageVO,
  type OrderQuery,
  type OrderStatus,
  type PayMethod,
} from "dehaze-sdk-js";
import { useHasPerm } from "@/hooks/usePermission";
import {
  Button,
  Card,
  DatePicker,
  Form,
  Input,
  InputNumber,
  Select,
  Space,
  Table,
  Tag,
  type TableColumnsType,
} from "antd";
import type { Dayjs } from "dayjs";
import {
  BarChartOutlined,
  EyeOutlined,
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
import OrderDetailDrawer, {
  type OrderDetailDrawerRef,
} from "./components/OrderDetailDrawer";
import OrderStatsDrawer, {
  type OrderStatsDrawerRef,
} from "./components/OrderStatsDrawer";
import "./index.scss";

const STATUS_OPTIONS: { label: string; value: OrderStatus }[] = [
  { label: "待支付", value: "pending" },
  { label: "已支付", value: "paid" },
  { label: "已完成", value: "completed" },
  { label: "已取消", value: "cancelled" },
  { label: "退款中", value: "refunding" },
  { label: "已退款", value: "refunded" },
];

const STATUS_MAP: Record<OrderStatus, { label: string; color: string }> = {
  pending: { label: "待支付", color: "warning" },
  paid: { label: "已支付", color: "processing" },
  completed: { label: "已完成", color: "default" },
  cancelled: { label: "已取消", color: "default" },
  refunding: { label: "退款中", color: "warning" },
  refunded: { label: "已退款", color: "default" },
};

const PAY_METHOD_OPTIONS: { label: string; value: PayMethod }[] = [
  { label: "微信支付", value: "wechat" },
  { label: "支付宝", value: "alipay" },
  { label: "余额支付", value: "balance" },
  { label: "组合支付", value: "combined" },
];

const PAY_METHOD_LABEL: Record<PayMethod, string> = {
  wechat: "微信支付",
  alipay: "支付宝",
  balance: "余额支付",
  combined: "组合支付",
};

interface SearchFormValues {
  orderNo?: string;
  keywords?: string;
  status?: OrderStatus;
  payMethod?: PayMethod;
  amountMin?: number;
  amountMax?: number;
  paidTimeRange?: [Dayjs, Dayjs];
}

const OrderManagement: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [pageData, setPageData] = useState<OrderPageVO[]>([]);
  const [total, setTotal] = useState(0);
  const [searchForm] = Form.useForm<SearchFormValues>();
  const [queryParams, setQueryParams] = useState<OrderQuery>({
    pageNum: 1,
    pageSize: 10,
  });

  const detailDrawerRef = useRef<OrderDetailDrawerRef>(null);
  const statsDrawerRef = useRef<OrderStatsDrawerRef>(null);
  const [refreshFlag, setRefreshFlag] = useState(0);

  const hasPerm = useHasPerm();

  const loadData = useCallback(async (params: OrderQuery) => {
    setLoading(true);
    try {
      const result = await OrderAPI.getPage(params);
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
    (values: SearchFormValues) => {
      setQueryParams({
        pageNum: 1,
        pageSize: queryParams.pageSize,
        orderNo: values.orderNo || undefined,
        keywords: values.keywords || undefined,
        status: values.status,
        payMethod: values.payMethod,
        amountMin: values.amountMin,
        amountMax: values.amountMax,
        paidTimeStart: values.paidTimeRange?.[0]?.format("YYYY-MM-DD HH:mm:ss"),
        paidTimeEnd: values.paidTimeRange?.[1]?.format("YYYY-MM-DD HH:mm:ss"),
      });
    },
    [queryParams.pageSize]
  );

  const handleReset = useCallback(() => {
    searchForm.resetFields();
    setQueryParams({ pageNum: 1, pageSize: 10 });
  }, [searchForm]);

  const handlePageChange = useCallback((page: number, pageSize: number) => {
    setQueryParams((prev) => ({ ...prev, pageNum: page, pageSize }));
  }, []);

  const handleDetail = useCallback((record: OrderPageVO) => {
    detailDrawerRef.current?.open(record.orderNo);
  }, []);

  const handleOpenStats = useCallback(() => {
    statsDrawerRef.current?.open();
  }, []);

  const columns: TableColumnsType<OrderPageVO> = useMemo(
    () => [
      {
        title: "订单号",
        dataIndex: "orderNo",
        key: "orderNo",
        width: 200,
        align: "center",
      },
      {
        title: "用户",
        dataIndex: "username",
        key: "username",
        width: 120,
        align: "center",
      },
      {
        title: "套餐",
        key: "package",
        minWidth: 160,
        render: (_: unknown, record: OrderPageVO) => (
          <Space>
            <span>{record.packageName}</span>
            <Tag color="default">{record.packageLevel}</Tag>
          </Space>
        ),
      },
      {
        title: "实付",
        dataIndex: "payableAmount",
        key: "payableAmount",
        width: 100,
        align: "right",
        render: (v: number) => `¥${(v ?? 0).toFixed(2)}`,
      },
      {
        title: "优惠",
        key: "discount",
        width: 120,
        align: "right",
        render: (_: unknown, record: OrderPageVO) => {
          const sum = (record.discountAmount ?? 0) + (record.couponAmount ?? 0);
          return sum > 0 ? `-¥${sum.toFixed(2)}` : "-";
        },
      },
      {
        title: "支付方式",
        dataIndex: "payMethod",
        key: "payMethod",
        width: 110,
        align: "center",
        render: (method?: PayMethod) =>
          method ? <Tag>{PAY_METHOD_LABEL[method]}</Tag> : "-",
      },
      {
        title: "状态",
        dataIndex: "status",
        key: "status",
        width: 100,
        align: "center",
        render: (status: OrderStatus) => (
          <Tag color={STATUS_MAP[status].color}>{STATUS_MAP[status].label}</Tag>
        ),
      },
      {
        title: "创建时间",
        dataIndex: "createTime",
        key: "createTime",
        width: 170,
        align: "center",
      },
      {
        title: "支付时间",
        dataIndex: "paidTime",
        key: "paidTime",
        width: 170,
        align: "center",
        render: (v?: string) => v || "-",
      },
      {
        title: "操作",
        key: "action",
        width: 110,
        align: "center",
        fixed: "right",
        render: (_: unknown, record: OrderPageVO) => (
          <Button
            type="link"
            size="small"
            icon={<EyeOutlined />}
            onClick={() => handleDetail(record)}
          >
            详情
          </Button>
        ),
      },
    ],
    [handleDetail]
  );

  return (
    <div className="order-management-container">
      <Card className="search-card" size="small">
        <Form form={searchForm} layout="inline" onFinish={handleSearch}>
          <Form.Item name="orderNo" label="订单号">
            <Input placeholder="订单号" allowClear style={{ width: 180 }} />
          </Form.Item>
          <Form.Item name="keywords" label="关键字">
            <Input
              placeholder="用户名/套餐"
              allowClear
              style={{ width: 180 }}
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
          <Form.Item name="payMethod" label="支付方式">
            <Select
              placeholder="全部"
              allowClear
              style={{ width: 120 }}
              options={PAY_METHOD_OPTIONS}
            />
          </Form.Item>
          <Form.Item label="金额区间">
            <Space>
              <Form.Item name="amountMin" noStyle>
                <InputNumber
                  min={0}
                  precision={2}
                  style={{ width: 110 }}
                  placeholder="最小"
                />
              </Form.Item>
              <span>-</span>
              <Form.Item name="amountMax" noStyle>
                <InputNumber
                  min={0}
                  precision={2}
                  style={{ width: 110 }}
                  placeholder="最大"
                />
              </Form.Item>
            </Space>
          </Form.Item>
          <Form.Item name="paidTimeRange" label="支付时间">
            <DatePicker.RangePicker
              showTime
              format="YYYY-MM-DD HH:mm:ss"
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
            </Space>
          </Form.Item>
        </Form>
      </Card>

      <Card
        className="table-card"
        size="small"
        title={
          hasPerm("order:stats") ? (
            <Button
              type="primary"
              icon={<BarChartOutlined />}
              onClick={handleOpenStats}
            >
              订单统计
            </Button>
          ) : undefined
        }
      >
        <Table
          columns={columns}
          dataSource={pageData}
          rowKey="orderNo"
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

      <OrderDetailDrawer ref={detailDrawerRef} />
      <OrderStatsDrawer ref={statsDrawerRef} />
    </div>
  );
};

export default OrderManagement;
