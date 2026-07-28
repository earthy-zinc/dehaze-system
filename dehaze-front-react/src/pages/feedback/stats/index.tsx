import * as echarts from "echarts";
import {
  FeedbackAPI,
  type FeedbackStatsVO,
  type FeedbackStatus,
  type FeedbackType,
  type RatingStatsVO,
} from "dehaze-sdk-js";
import {
  Card,
  Col,
  DatePicker,
  Empty,
  Progress,
  Rate,
  Row,
  Spin,
  Table,
  Tabs,
  Tag,
  type TableColumnsType,
} from "antd";
import {
  BarChartOutlined,
  MessageOutlined,
  StarOutlined,
} from "@ant-design/icons";
import type { Dayjs } from "dayjs";
import dayjs from "dayjs";
import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useSearchParams } from "react-router-dom";
import "./index.scss";

const { RangePicker } = DatePicker;

const RATING_COLOR_MAP: Record<number, string> = {
  1: "#f5222d",
  2: "#fa8c16",
  3: "#faad14",
  4: "#409eff",
  5: "#52c41a",
};

const TYPE_LABEL: Record<FeedbackType, string> = {
  suggestion: "功能建议",
  bug: "问题报告",
  experience: "体验反馈",
  complaint: "投诉",
};
const TYPE_COLOR: Record<FeedbackType, string> = {
  suggestion: "#409eff",
  experience: "#67c23a",
  complaint: "#e6a23c",
  bug: "#f56c6c",
};

const STATUS_LABEL: Record<FeedbackStatus, string> = {
  pending: "待处理",
  processing: "处理中",
  replied: "已回复",
  closed: "已关闭",
};
const STATUS_COLOR: Record<FeedbackStatus, string> = {
  pending: "#e6a23c",
  processing: "#409eff",
  replied: "#67c23a",
  closed: "#909399",
};

function pad(n: number) {
  return String(n).padStart(2, "0");
}
function formatDate(d: Dayjs) {
  return `${d.year()}-${pad(d.month() + 1)}-${pad(d.date())}`;
}
function defaultRange(): [Dayjs, Dayjs] {
  const end = dayjs();
  const start = end.subtract(30, "day");
  return [start, end];
}

const FeedbackStats: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const activeTab =
    searchParams.get("tab") === "feedback" ? "feedback" : "rating";

  const [timeRange, setTimeRange] = useState<[Dayjs, Dayjs]>(defaultRange());
  const [ratingLoading, setRatingLoading] = useState(false);
  const [feedbackLoading, setFeedbackLoading] = useState(false);
  const [ratingStats, setRatingStats] = useState<RatingStatsVO | null>(null);
  const [feedbackStats, setFeedbackStats] = useState<FeedbackStatsVO | null>(
    null
  );

  const ratingDistRef = useRef<HTMLDivElement>(null);
  const feedbackTypeRef = useRef<HTMLDivElement>(null);
  const feedbackStatusRef = useRef<HTMLDivElement>(null);

  const ratingDistChart = useRef<echarts.ECharts | null>(null);
  const feedbackTypeChart = useRef<echarts.ECharts | null>(null);
  const feedbackStatusChart = useRef<echarts.ECharts | null>(null);

  const startTime = timeRange[0]?.format("YYYY-MM-DD");
  const endTime = timeRange[1]?.format("YYYY-MM-DD");

  const loadRatingStats = useCallback(async () => {
    setRatingLoading(true);
    try {
      const data = await FeedbackAPI.getRatingStats(startTime, endTime);
      setRatingStats(data);
    } finally {
      setRatingLoading(false);
    }
  }, [startTime, endTime]);

  const loadFeedbackStats = useCallback(async () => {
    setFeedbackLoading(true);
    try {
      const data = await FeedbackAPI.getFeedbackStats(startTime, endTime);
      setFeedbackStats(data);
    } finally {
      setFeedbackLoading(false);
    }
  }, [startTime, endTime]);

  useEffect(() => {
    loadRatingStats();
    loadFeedbackStats();
  }, [loadRatingStats, loadFeedbackStats]);

  const renderRatingDistChart = useCallback((data: RatingStatsVO) => {
    if (!ratingDistRef.current) return;
    if (ratingDistChart.current) {
      ratingDistChart.current.dispose();
    }
    ratingDistChart.current = echarts.init(ratingDistRef.current);
    ratingDistChart.current.setOption({
      tooltip: { trigger: "item", formatter: "{b}: {c} ({d}%)" },
      legend: { bottom: 0 },
      series: [
        {
          type: "pie",
          radius: ["40%", "70%"],
          label: { formatter: "{b}: {c}" },
          data: Object.entries(data.ratingDistribution || {}).map(([k, v]) => ({
            name: `${k}星`,
            value: v,
            itemStyle: { color: RATING_COLOR_MAP[Number(k)] },
          })),
        },
      ],
    });
  }, []);

  const renderFeedbackCharts = useCallback((data: FeedbackStatsVO) => {
    if (feedbackTypeRef.current) {
      if (feedbackTypeChart.current) {
        feedbackTypeChart.current.dispose();
      }
      feedbackTypeChart.current = echarts.init(feedbackTypeRef.current);
      feedbackTypeChart.current.setOption({
        tooltip: { trigger: "item", formatter: "{b}: {c} ({d}%)" },
        legend: { bottom: 0 },
        series: [
          {
            type: "pie",
            radius: ["40%", "70%"],
            label: { formatter: "{b}: {c}" },
            data: (Object.keys(TYPE_LABEL) as FeedbackType[]).map((t) => ({
              name: TYPE_LABEL[t],
              value: data.typeDistribution?.[t] || 0,
              itemStyle: { color: TYPE_COLOR[t] },
            })),
          },
        ],
      });
    }

    if (feedbackStatusRef.current) {
      if (feedbackStatusChart.current) {
        feedbackStatusChart.current.dispose();
      }
      feedbackStatusChart.current = echarts.init(feedbackStatusRef.current);
      const statuses = Object.keys(STATUS_LABEL) as FeedbackStatus[];
      feedbackStatusChart.current.setOption({
        tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
        legend: { bottom: 0 },
        grid: { left: "3%", right: "3%", bottom: "15%", containLabel: true },
        xAxis: {
          type: "category",
          data: statuses.map((s) => STATUS_LABEL[s]),
        },
        yAxis: { type: "value" },
        series: [
          {
            type: "bar",
            barWidth: "40%",
            data: statuses.map((s) => ({
              value: data.statusDistribution?.[s] || 0,
              itemStyle: { color: STATUS_COLOR[s] },
            })),
          },
        ],
      });
    }
  }, []);

  useEffect(() => {
    if (activeTab === "rating" && ratingStats) {
      renderRatingDistChart(ratingStats);
    } else if (activeTab === "feedback" && feedbackStats) {
      renderFeedbackCharts(feedbackStats);
    }
    return () => {
      if (activeTab === "rating") {
        ratingDistChart.current?.dispose();
        ratingDistChart.current = null;
      } else {
        feedbackTypeChart.current?.dispose();
        feedbackTypeChart.current = null;
        feedbackStatusChart.current?.dispose();
        feedbackStatusChart.current = null;
      }
    };
  }, [
    activeTab,
    ratingStats,
    feedbackStats,
    renderRatingDistChart,
    renderFeedbackCharts,
  ]);

  useEffect(() => {
    const handleResize = () => {
      ratingDistChart.current?.resize();
      feedbackTypeChart.current?.resize();
      feedbackStatusChart.current?.resize();
    };
    window.addEventListener("resize", handleResize);
    return () => {
      window.removeEventListener("resize", handleResize);
      ratingDistChart.current?.dispose();
      feedbackTypeChart.current?.dispose();
      feedbackStatusChart.current?.dispose();
    };
  }, []);

  const handleTabChange = (key: string) => {
    setSearchParams({ tab: key });
  };

  const handleTimeChange = (values: [Dayjs, Dayjs] | null) => {
    if (!values || values.length !== 2) return;
    setTimeRange(values);
  };

  const ratingMetrics = useMemo(() => {
    const dist = ratingStats?.ratingDistribution || {};
    const total = ratingStats?.totalRatings || 0;
    const good = (dist[4] || 0) + (dist[5] || 0);
    const bad = (dist[1] || 0) + (dist[2] || 0);
    return {
      total,
      average: ratingStats?.averageRating || 0,
      positiveRate: total ? Math.round((good * 10000) / total) / 100 : 0,
      negativeRate: total ? Math.round((bad * 10000) / total) / 100 : 0,
    };
  }, [ratingStats]);

  const feedbackMetrics = useMemo(() => {
    const total = feedbackStats?.totalFeedback || 0;
    const pending = feedbackStats?.statusDistribution?.pending || 0;
    return {
      total,
      avgResponse: feedbackStats?.averageResponseTime ?? 0,
      avgClose: feedbackStats?.averageCloseTime ?? 0,
      pending,
    };
  }, [feedbackStats]);

  const positiveMax = useMemo(
    () =>
      Math.max(
        1,
        ...(ratingStats?.positiveTagRanking || []).map((t) => t.count)
      ),
    [ratingStats]
  );
  const negativeMax = useMemo(
    () =>
      Math.max(
        1,
        ...(ratingStats?.negativeTagRanking || []).map((t) => t.count)
      ),
    [ratingStats]
  );
  const keywordMax = useMemo(
    () =>
      Math.max(1, ...(feedbackStats?.topKeywords || []).map((k) => k.count)),
    [feedbackStats]
  );
  const keywordFontSize = (count: number) =>
    Math.round(14 + (count / keywordMax) * 10);

  const algorithmColumns: TableColumnsType<
    RatingStatsVO["algorithmStats"][number]
  > = [
    {
      title: "算法",
      dataIndex: "algorithmName",
      key: "algorithmName",
      minWidth: 160,
    },
    {
      title: "平均评分",
      dataIndex: "averageRating",
      key: "averageRating",
      width: 180,
      align: "center",
      render: (val: number) => <Rate disabled allowHalf value={val} />,
    },
    {
      title: "评价数",
      dataIndex: "totalRatings",
      key: "totalRatings",
      width: 100,
      align: "center",
    },
    {
      title: "差评率",
      dataIndex: "lowRatingRate",
      key: "lowRatingRate",
      minWidth: 220,
      render: (val: number) => (
        <Progress
          percent={val}
          strokeColor={val > 20 ? "#f5222d" : "#409eff"}
        />
      ),
    },
  ];

  const tagColumn = (max: number, color: string) =>
    ({
      title: "相对占比",
      key: "percent",
      minWidth: 180,
      render: (_: unknown, record: { count: number }) => (
        <Progress
          percent={Math.round((record.count * 100) / max)}
          strokeColor={color}
        />
      ),
    }) as TableColumnsType<{ tag: string; count: number }>[number];

  const tagColumns = (max: number, color: string) =>
    [
      { title: "标签", dataIndex: "tag", key: "tag", minWidth: 120 },
      {
        title: "次数",
        dataIndex: "count",
        key: "count",
        width: 80,
        align: "center",
      },
      tagColumn(max, color),
    ] as TableColumnsType<{ tag: string; count: number }>;

  const moduleColumns: TableColumnsType<{
    module: string;
    count: number;
  }> = [
    { title: "模块", dataIndex: "module", key: "module", minWidth: 160 },
    {
      title: "反馈数",
      dataIndex: "count",
      key: "count",
      width: 100,
      align: "center",
    },
    {
      title: "占比",
      key: "percent",
      minWidth: 240,
      render: (_: unknown, record: { count: number }) => (
        <Progress
          percent={
            feedbackMetrics.total
              ? Math.round((record.count * 100) / feedbackMetrics.total)
              : 0
          }
        />
      ),
    },
  ];

  return (
    <div className="feedback-stats-container">
      <Card className="filter-card" size="small">
        <span className="filter-label">时间范围：</span>
        <RangePicker
          value={timeRange}
          onChange={(values) =>
            handleTimeChange(values as [Dayjs, Dayjs] | null)
          }
          allowClear={false}
          style={{ width: 260 }}
        />
      </Card>

      <Tabs
        activeKey={activeTab}
        onChange={handleTabChange}
        items={[
          {
            key: "rating",
            label: (
              <span>
                <StarOutlined /> 评价统计
              </span>
            ),
            children: (
              <Spin spinning={ratingLoading}>
                <Row gutter={16} className="stat-cards">
                  <Col span={6}>
                    <Card hoverable>
                      <div className="stat-card">
                        <div className="stat-label">总评价数</div>
                        <div className="stat-value">{ratingMetrics.total}</div>
                      </div>
                    </Card>
                  </Col>
                  <Col span={6}>
                    <Card hoverable>
                      <div className="stat-card">
                        <div className="stat-label">平均评分</div>
                        <div className="stat-value">
                          {ratingMetrics.average.toFixed(2)}
                        </div>
                        <Rate
                          disabled
                          allowHalf
                          value={ratingMetrics.average}
                        />
                      </div>
                    </Card>
                  </Col>
                  <Col span={6}>
                    <Card hoverable>
                      <div className="stat-card">
                        <div className="stat-label">好评率</div>
                        <div
                          className="stat-value"
                          style={{ color: "#52c41a" }}
                        >
                          {ratingMetrics.positiveRate}%
                        </div>
                      </div>
                    </Card>
                  </Col>
                  <Col span={6}>
                    <Card hoverable>
                      <div className="stat-card">
                        <div className="stat-label">差评率</div>
                        <div
                          className="stat-value"
                          style={{ color: "#f5222d" }}
                        >
                          {ratingMetrics.negativeRate}%
                        </div>
                      </div>
                    </Card>
                  </Col>
                </Row>

                <Card
                  size="small"
                  className="section-card"
                  title={
                    <span className="card-title">
                      <BarChartOutlined /> 评分分布
                    </span>
                  }
                >
                  <div
                    ref={ratingDistRef}
                    style={{ width: "100%", height: 300 }}
                  />
                </Card>

                <Card
                  size="small"
                  className="section-card"
                  title={
                    <span className="card-title">
                      <BarChartOutlined /> 算法维度统计
                    </span>
                  }
                >
                  <Table
                    size="small"
                    columns={algorithmColumns}
                    dataSource={ratingStats?.algorithmStats || []}
                    rowKey={(record) => record.algorithmId}
                    pagination={false}
                    bordered
                  />
                </Card>

                <Row gutter={16} className="section-card">
                  <Col span={12}>
                    <Card size="small" title="正面标签排行">
                      <Table
                        size="small"
                        columns={tagColumns(positiveMax, "#52c41a")}
                        dataSource={ratingStats?.positiveTagRanking || []}
                        rowKey="tag"
                        pagination={false}
                        bordered
                      />
                    </Card>
                  </Col>
                  <Col span={12}>
                    <Card size="small" title="负面标签排行">
                      <Table
                        size="small"
                        columns={tagColumns(negativeMax, "#f5222d")}
                        dataSource={ratingStats?.negativeTagRanking || []}
                        rowKey="tag"
                        pagination={false}
                        bordered
                      />
                    </Card>
                  </Col>
                </Row>
              </Spin>
            ),
          },
          {
            key: "feedback",
            label: (
              <span>
                <MessageOutlined /> 反馈统计
              </span>
            ),
            children: (
              <Spin spinning={feedbackLoading}>
                <Row gutter={16} className="stat-cards">
                  <Col span={6}>
                    <Card hoverable>
                      <div className="stat-card">
                        <div className="stat-label">总反馈数</div>
                        <div className="stat-value">
                          {feedbackMetrics.total}
                        </div>
                      </div>
                    </Card>
                  </Col>
                  <Col span={6}>
                    <Card hoverable>
                      <div className="stat-card">
                        <div className="stat-label">平均响应时间</div>
                        <div className="stat-value">
                          {feedbackMetrics.avgResponse}
                          <span className="unit">分钟</span>
                        </div>
                      </div>
                    </Card>
                  </Col>
                  <Col span={6}>
                    <Card hoverable>
                      <div className="stat-card">
                        <div className="stat-label">平均关闭时间</div>
                        <div className="stat-value">
                          {feedbackMetrics.avgClose}
                          <span className="unit">小时</span>
                        </div>
                      </div>
                    </Card>
                  </Col>
                  <Col span={6}>
                    <Card hoverable>
                      <div className="stat-card">
                        <div className="stat-label">待处理数</div>
                        <div
                          className="stat-value"
                          style={{ color: "#e6a23c" }}
                        >
                          {feedbackMetrics.pending}
                        </div>
                      </div>
                    </Card>
                  </Col>
                </Row>

                <Row gutter={16} className="section-card">
                  <Col span={12}>
                    <Card
                      size="small"
                      title={
                        <span className="card-title">
                          <BarChartOutlined /> 类型分布
                        </span>
                      }
                    >
                      <div
                        ref={feedbackTypeRef}
                        style={{ width: "100%", height: 300 }}
                      />
                    </Card>
                  </Col>
                  <Col span={12}>
                    <Card
                      size="small"
                      title={
                        <span className="card-title">
                          <BarChartOutlined /> 状态分布
                        </span>
                      }
                    >
                      <div
                        ref={feedbackStatusRef}
                        style={{ width: "100%", height: 300 }}
                      />
                    </Card>
                  </Col>
                </Row>

                <Card
                  size="small"
                  className="section-card"
                  title={
                    <span className="card-title">
                      <BarChartOutlined /> 模块分布
                    </span>
                  }
                >
                  <Table
                    size="small"
                    columns={moduleColumns}
                    dataSource={feedbackStats?.moduleDistribution || []}
                    rowKey="module"
                    pagination={false}
                    bordered
                  />
                </Card>

                <Card
                  size="small"
                  className="section-card"
                  title={
                    <span className="card-title">
                      <BarChartOutlined /> 高频关键词
                    </span>
                  }
                >
                  <div className="keyword-cloud">
                    {feedbackStats?.topKeywords?.length ? (
                      feedbackStats.topKeywords.map((kw) => (
                        <Tag
                          key={kw.keyword}
                          style={{
                            fontSize: keywordFontSize(kw.count),
                            margin: 6,
                          }}
                        >
                          {kw.keyword} ({kw.count})
                        </Tag>
                      ))
                    ) : (
                      <Empty description="暂无数据" />
                    )}
                  </div>
                </Card>
              </Spin>
            ),
          },
        ]}
      />
    </div>
  );
};

export default FeedbackStats;
