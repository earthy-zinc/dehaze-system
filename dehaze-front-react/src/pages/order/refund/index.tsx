import {
  OrderAPI,
  type RefundQuery,
  type RefundRecordVO,
  type RefundStatus,
} from "dehaze-sdk-js";
import { useHasPerm } from "@/hooks/usePermission";
import {
  Button,
  Card,
  DatePicker,
  Form,
  Input,
  Select,
  Space,
  Table,
  Tag,
  type TableColumnsType,
} from "antd";
import type { Dayjs } from "dayjs";
import {
  CheckOutlined,
  CloseOutlined,
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
import RefundAuditDialog, {
  type RefundAuditDialogRef,
} from "./components/RefundAuditDialog";
import "./index.scss";

const STATUS_OPTIONS: { label: string; value: RefundStatus }[] = [
  { label: "退款中", value: "refunding" },
  { label: "退款成功", value: "refunded" },
  { label: "退款失败", value: "refund_failed" },
];

const REFUND_STATUS_MAP: Record<
  RefundStatus,
  { label: string; color: string }
> = {
  refunding: { label: "退款中", color: "warning" },
  refunded: { label: "退款成功", color: "default" },
  refund_failed: { label: "退款失败", color: "error" },
};

interface SearchFormValues {
  orderNo?: string;
  keywords?: string;
  status?: RefundStatus;
  applyTimeRange?: [Dayjs, Dayjs];
}

const RefundManagement: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [pageData, setPageData] = useState<RefundRecordVO[]>([]);
  const [total, setTotal] = useState(0);
  const [searchForm] = Form.useForm<SearchFormValues>();
  const [queryParams, setQueryParams] = useState<RefundQuery>({
    pageNum: 1,
    pageSize: 10,
  });

  const auditDialogRef = useRef<RefundAuditDialogRef>(null);
  const [refreshFlag, setRefreshFlag] = useState(0);

  const hasPerm = useHasPerm();

  const loadData = useCallback(async (params: RefundQuery) => {
    setLoading(true);
    try {
      const result = await OrderAPI.listRefunds(params);
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
        applyTimeStart: values.applyTimeRange?.[0]?.format(
          "YYYY-MM-DD HH:mm:ss"
        ),
        applyTimeEnd: values.applyTimeRange?.[1]?.format("YYYY-MM-DD HH:mm:ss"),
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

  const handleAudit = useCallback(
    (record: RefundRecordVO, approved: boolean) => {
      auditDialogRef.current?.open(record, approved);
    },
    []
  );

  const columns: TableColumnsType<RefundRecordVO> = useMemo(
    () => [
      {
        title: "退款单号",
        dataIndex: "refundNo",
        key: "refundNo",
        width: 200,
        align: "center",
      },
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
        title: "退款金额",
        dataIndex: "refundAmount",
        key: "refundAmount",
        width: 110,
        align: "right",
        render: (v: number) => `¥${v.toFixed(2)}`,
      },
      {
        title: "退款原因",
        dataIndex: "reason",
        key: "reason",
        minWidth: 160,
        ellipsis: true,
      },
      {
        title: "已用配额",
        dataIndex: "usedQuota",
        key: "usedQuota",
        width: 100,
        align: "center",
      },
      {
        title: "状态",
        dataIndex: "status",
        key: "status",
        width: 110,
        align: "center",
        render: (status: RefundStatus) => (
          <Tag color={REFUND_STATUS_MAP[status].color}>
            {REFUND_STATUS_MAP[status].label}
          </Tag>
        ),
      },
      {
        title: "申请时间",
        dataIndex: "applyTime",
        key: "applyTime",
        width: 170,
        align: "center",
      },
      {
        title: "审核时间",
        dataIndex: "auditTime",
        key: "auditTime",
        width: 170,
        align: "center",
        render: (v?: string) => v || "-",
      },
      {
        title: "操作",
        key: "action",
        width: 160,
        align: "center",
        fixed: "right",
        render: (_: unknown, record: RefundRecordVO) =>
          record.status === "refunding" ? (
            <Space size="small">
              {hasPerm("order:refund:audit") && (
                <Button
                  type="link"
                  size="small"
                  icon={<CheckOutlined />}
                  onClick={() => handleAudit(record, true)}
                >
                  通过
                </Button>
              )}
              {hasPerm("order:refund:audit") && (
                <Button
                  type="link"
                  size="small"
                  danger
                  icon={<CloseOutlined />}
                  onClick={() => handleAudit(record, false)}
                >
                  驳回
                </Button>
              )}
            </Space>
          ) : (
            "-"
          ),
      },
    ],
    [handleAudit, hasPerm]
  );

  return (
    <div className="refund-management-container">
      <Card className="search-card" size="small">
        <Form form={searchForm} layout="inline" onFinish={handleSearch}>
          <Form.Item name="orderNo" label="订单号">
            <Input placeholder="订单号" allowClear style={{ width: 180 }} />
          </Form.Item>
          <Form.Item name="keywords" label="关键字">
            <Input
              placeholder="退款单号/用户名"
              allowClear
              style={{ width: 180 }}
            />
          </Form.Item>
          <Form.Item name="status" label="退款状态">
            <Select
              placeholder="全部"
              allowClear
              style={{ width: 140 }}
              options={STATUS_OPTIONS}
            />
          </Form.Item>
          <Form.Item name="applyTimeRange" label="申请时间">
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

      <Card className="table-card" size="small">
        <Table
          columns={columns}
          dataSource={pageData}
          rowKey="refundNo"
          loading={loading}
          scroll={{ x: 1400 }}
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

      <RefundAuditDialog ref={auditDialogRef} onSuccess={refreshList} />
    </div>
  );
};

export default RefundManagement;
