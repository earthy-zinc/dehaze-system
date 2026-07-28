import {
  MemberAPI,
  type GrowthChangeType,
  type GrowthLogVO,
  type GrowthLogQuery,
} from "dehaze-sdk-js";
import {
  Button,
  Card,
  DatePicker,
  Form,
  Select,
  Space,
  Table,
  type TableColumnsType,
} from "antd";
import {
  ArrowLeftOutlined,
  ReloadOutlined,
  SearchOutlined,
} from "@ant-design/icons";
import React, { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import "./index.scss";

const { RangePicker } = DatePicker;

const CHANGE_TYPE_OPTIONS: { value: GrowthChangeType; label: string }[] = [
  { value: "dehaze", label: "去雾处理" },
  { value: "evaluate", label: "指标评估" },
  { value: "rating", label: "评价奖励" },
  { value: "sign_in", label: "每日签到" },
  { value: "sign_in_bonus", label: "签到奖励" },
  { value: "consume", label: "消费获得" },
  { value: "refund_deduct", label: "退款扣除" },
  { value: "admin_adjust", label: "后台调整" },
];

const CHANGE_TYPE_LABEL: Record<string, string> = Object.fromEntries(
  CHANGE_TYPE_OPTIONS.map((o) => [o.value, o.label])
);

const GrowthLogs: React.FC = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [pageData, setPageData] = useState<GrowthLogVO[]>([]);
  const [total, setTotal] = useState(0);
  const [searchForm] = Form.useForm();
  const [queryParams, setQueryParams] = useState<GrowthLogQuery>({
    pageNum: 1,
    pageSize: 20,
  });
  const [refreshFlag, setRefreshFlag] = useState(0);

  const loadData = useCallback(async (params: GrowthLogQuery) => {
    setLoading(true);
    try {
      const result = await MemberAPI.getGrowthLogs(params);
      setPageData(result.list || []);
      setTotal(result.total || 0);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData(queryParams);
  }, [queryParams, refreshFlag]);

  const handleSearch = useCallback(
    (values: {
      changeType?: GrowthChangeType;
      timeRange?: [string, string];
    }) => {
      setQueryParams((prev) => ({
        ...prev,
        pageNum: 1,
        changeType: values.changeType,
        startTime: values.timeRange?.[0]
          ? `${values.timeRange[0]} 00:00:00`
          : undefined,
        endTime: values.timeRange?.[1]
          ? `${values.timeRange[1]} 23:59:59`
          : undefined,
      }));
    },
    []
  );

  const handleReset = useCallback(() => {
    searchForm.resetFields();
    setQueryParams({ pageNum: 1, pageSize: 20 });
  }, [searchForm]);

  const handlePageChange = useCallback((page: number, pageSize: number) => {
    setQueryParams((prev) => ({ ...prev, pageNum: page, pageSize }));
  }, []);

  const columns: TableColumnsType<GrowthLogVO> = useMemo(
    () => [
      {
        title: "#",
        key: "index",
        width: 50,
        align: "center",
        render: (_: unknown, __: GrowthLogVO, index: number) => index + 1,
      },
      {
        title: "变动类型",
        dataIndex: "changeType",
        key: "changeType",
        width: 130,
        align: "center",
        render: (type: GrowthChangeType) => (
          <span className={`type-tag tag-${type}`}>
            {CHANGE_TYPE_LABEL[type] || type}
          </span>
        ),
      },
      {
        title: "变动值",
        dataIndex: "changeValue",
        key: "changeValue",
        width: 100,
        align: "center",
        render: (value: number) => (
          <span className={value >= 0 ? "value-up" : "value-down"}>
            {value >= 0 ? "+" : ""}
            {value}
          </span>
        ),
      },
      {
        title: "变动后余额",
        dataIndex: "balance",
        key: "balance",
        width: 120,
        align: "center",
      },
      {
        title: "原因",
        dataIndex: "reason",
        key: "reason",
        ellipsis: true,
        render: (reason?: string) => reason || "-",
      },
      {
        title: "关联ID",
        dataIndex: "relatedId",
        key: "relatedId",
        width: 120,
        align: "center",
        render: (relatedId?: string) => relatedId || "-",
      },
      {
        title: "操作人",
        dataIndex: "operatorId",
        key: "operatorId",
        width: 100,
        align: "center",
        render: (operatorId?: number) =>
          operatorId ? `用户${operatorId}` : "系统",
      },
      {
        title: "时间",
        dataIndex: "createTime",
        key: "createTime",
        width: 180,
        align: "center",
      },
    ],
    []
  );

  return (
    <div className="growth-logs">
      <div className="page-header">
        <div className="header-title">
          <Button
            type="link"
            icon={<ArrowLeftOutlined />}
            onClick={() => navigate("/member/center")}
          />
          <span className="title-text">成长值明细</span>
        </div>
      </div>

      <Card className="search-card" size="small">
        <Form form={searchForm} layout="inline" onFinish={handleSearch}>
          <Form.Item name="changeType" label="变动类型">
            <Select
              placeholder="全部"
              allowClear
              style={{ width: 150 }}
              options={CHANGE_TYPE_OPTIONS}
            />
          </Form.Item>
          <Form.Item name="timeRange" label="时间范围">
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
        <Table
          columns={columns}
          dataSource={pageData}
          rowKey="id"
          loading={loading}
          scroll={{ x: 1000 }}
          pagination={{
            current: queryParams.pageNum,
            pageSize: queryParams.pageSize,
            total,
            showSizeChanger: true,
            pageSizeOptions: ["10", "20", "50", "100"],
            showTotal: (t) => `共 ${t} 条`,
            onChange: handlePageChange,
          }}
        />
      </Card>
    </div>
  );
};

export default GrowthLogs;
