import { AlgorithmAPI, type Algorithm } from "dehaze-sdk-js";
import {
  ArrowLeftOutlined,
  ClockCircleOutlined,
  FireOutlined,
  GlobalOutlined,
} from "@ant-design/icons";
import {
  Button,
  Card,
  Col,
  Descriptions,
  Empty,
  Image,
  Rate,
  Result,
  Row,
  Space,
  Spin,
  Statistic,
  Tabs,
  Tag,
  Typography,
  message,
} from "antd";
import React, { useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";

const { Title, Text, Paragraph } = Typography;

/** 算法状态映射 */
const STATUS_MAP: Record<number, { label: string; color: string }> = {
  0: { label: "待审核", color: "orange" },
  1: { label: "已启用", color: "green" },
  2: { label: "已禁用", color: "default" },
  3: { label: "已驳回", color: "red" },
  4: { label: "审核中", color: "blue" },
  5: { label: "已归档", color: "gray" },
};

/** 类型标签颜色映射 */
const TYPE_COLOR_MAP: Record<string, string> = {
  traditional: "blue",
  deep_learning: "volcano",
  hybrid: "cyan",
};

export default function AlgorithmDetail(): React.JSX.Element {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const id = Number(searchParams.get("id"));

  const [detailLoading, setDetailLoading] = useState(false);
  const [algorithm, setAlgorithm] = useState<Algorithm | null>(null);
  const [monitorData, setMonitorData] = useState<{
    callCount: number;
    avgTime: number;
    successRate: number;
  } | null>(null);

  // ==================== 路由参数校验 ====================

  const isValidId = searchParams.has("id") && !isNaN(id) && id > 0;

  // ==================== 数据加载 ====================

  useEffect(() => {
    if (!isValidId) return;
    let cancelled = false;
    const loadDetail = async () => {
      setDetailLoading(true);
      try {
        const data = await AlgorithmAPI.getAlgorithmInfoById(id);
        if (!cancelled) setAlgorithm(data);
      } catch (error: any) {
        if (!cancelled) message.error(error?.message || "获取算法详情失败");
      } finally {
        if (!cancelled) setDetailLoading(false);
      }
    };
    loadDetail();
  }, [id]);

  useEffect(() => {
    const loadMonitor = async () => {
      try {
        const result = await AlgorithmAPI.getMonitorData(id);
        if (result) setMonitorData(result);
      } catch {
        // Ignore monitor loading errors
      }
    };
    loadMonitor();
  }, [id]);

  // ==================== 派生数据 ====================

  const statusInfo =
    algorithm?.status != null ? STATUS_MAP[algorithm.status] : null;

  // ==================== 渲染 ====================

  if (detailLoading) {
    return (
      <div style={{ padding: 48, textAlign: "center" }}>
        <Spin tip="加载中..." size="large" />
      </div>
    );
  }

  if (!algorithm) {
    return (
      <div style={{ padding: 48, textAlign: "center" }}>
        <Empty description="未找到该算法信息" style={{ minHeight: 300 }}>
          <Button type="primary" onClick={() => navigate("/algorithm")}>
            返回算法列表
          </Button>
        </Empty>
      </div>
    );
  }

  /** Tab 1: 基本信息 */
  const basicTab = {
    key: "basic",
    label: "基本信息",
    children: (
      <Card size="small">
        <Descriptions bordered column={2} size="middle">
          <Descriptions.Item label="算法名称" span={2}>
            <Title level={4} style={{ margin: 0 }}>
              {algorithm.name}
            </Title>
          </Descriptions.Item>
          <Descriptions.Item label="算法类型">
            <Tag color={TYPE_COLOR_MAP[algorithm.type] || "blue"}>
              {algorithm.type}
            </Tag>
          </Descriptions.Item>
          <Descriptions.Item label="状态">
            <Tag color={statusInfo?.color || "default"}>
              {statusInfo?.label || `未知(${algorithm.status})`}
            </Tag>
          </Descriptions.Item>
          <Descriptions.Item label="版本号">
            <Tag>{algorithm.version || "-"}</Tag>
          </Descriptions.Item>
          <Descriptions.Item label="父节点">
            {algorithm.parentId ? String(algorithm.parentId) : "-"}
          </Descriptions.Item>
          <Descriptions.Item label="创建时间">
            {algorithm.createTime || "-"}
          </Descriptions.Item>
          <Descriptions.Item label="更新时间" span={2}>
            {algorithm.auditTime || "-"}
          </Descriptions.Item>
          <Descriptions.Item label="审核人" span={2}>
            {algorithm.auditBy || "-"}
          </Descriptions.Item>
          <Descriptions.Item label="审核备注" span={2}>
            {algorithm.auditRemark || "-"}
          </Descriptions.Item>
          <Descriptions.Item label="描述" span={2}>
            <Paragraph ellipsis={{ rows: 4 }} style={{ marginBottom: 0 }}>
              {algorithm.description || "暂无描述"}
            </Paragraph>
          </Descriptions.Item>
          {algorithm.path && (
            <Descriptions.Item label="路径" span={2}>
              <Text type="secondary" copyable>
                {algorithm.path}
              </Text>
            </Descriptions.Item>
          )}
          {algorithm.importPath && (
            <Descriptions.Item label="导入路径" span={2}>
              <Text type="secondary" copyable>
                {algorithm.importPath}
              </Text>
            </Descriptions.Item>
          )}
          {algorithm.params && (
            <Descriptions.Item label="参数配置" span={2}>
              <Text type="secondary" style={{ fontFamily: "monospace" }}>
                {algorithm.params}
              </Text>
            </Descriptions.Item>
          )}
        </Descriptions>
      </Card>
    ),
  };

  /** Tab 2: 技术信息 */
  const techTab = {
    key: "tech",
    label: "技术信息",
    children: (
      <Card size="small">
        <Descriptions bordered column={2} size="middle">
          <Descriptions.Item label="参数量">
            <Text>{algorithm.params || "-"}</Text>
          </Descriptions.Item>
          <Descriptions.Item label="计算量(FLOPS)">
            <Text>{algorithm.flops || "-"}</Text>
          </Descriptions.Item>
          <Descriptions.Item label="模型大小">
            <Text>{algorithm.size || "-"}</Text>
          </Descriptions.Item>
          <Descriptions.Item label="网络结构图" span={2}>
            {algorithm.img ? (
              <Image
                src={algorithm.img}
                alt="网络结构图"
                style={{ maxWidth: "100%", maxHeight: 400 }}
                fallback="https://via.placeholder.com/600x400?text=No+Architecture+Image"
              />
            ) : (
              <Empty
                description="暂无网络结构图"
                image={Empty.PRESENTED_IMAGE_SIMPLE}
              />
            )}
          </Descriptions.Item>
          {algorithm.path && (
            <Descriptions.Item label="模型路径" span={2}>
              <Text type="secondary" copyable>
                {algorithm.path}
              </Text>
            </Descriptions.Item>
          )}
          {algorithm.importPath && (
            <Descriptions.Item label="导入路径" span={2}>
              <Text type="secondary" copyable>
                {algorithm.importPath}
              </Text>
            </Descriptions.Item>
          )}
        </Descriptions>
      </Card>
    ),
  };

  /** Tab 3: 运营信息 */
  const opsTab = {
    key: "ops",
    label: "运营信息",
    children: (
      <Card size="small">
        <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
          <Col span={8}>
            <Card size="small">
              <Statistic
                title="调用次数"
                value={monitorData?.callCount ?? 0}
                prefix={<FireOutlined />}
                suffix="次"
              />
            </Card>
          </Col>
          <Col span={8}>
            <Card size="small">
              <Statistic
                title="平均耗时"
                value={monitorData?.avgTime ?? 0}
                prefix={<ClockCircleOutlined />}
                suffix="ms"
              />
            </Card>
          </Col>
          <Col span={8}>
            <Card size="small">
              <Statistic
                title="成功率"
                value={monitorData?.successRate ?? 0}
                prefix={<GlobalOutlined />}
                suffix="%"
                valueStyle={{
                  color:
                    (monitorData?.successRate ?? 0) >= 95
                      ? "#52c41a"
                      : (monitorData?.successRate ?? 0) >= 80
                        ? "#faad14"
                        : "#f5222d",
                }}
              />
            </Card>
          </Col>
        </Row>
        <Card size="small">
          <Descriptions bordered column={2} size="middle">
            <Descriptions.Item label="评分">
              <Rate
                disabled
                defaultValue={(monitorData?.successRate ?? 0) / 20}
              />
            </Descriptions.Item>
            <Descriptions.Item label="调用频率">
              {monitorData?.callCount ?? 0} 次
            </Descriptions.Item>
            <Descriptions.Item label="平均响应时间">
              {monitorData?.avgTime ?? 0} ms
            </Descriptions.Item>
            <Descriptions.Item label="成功率">
              {monitorData?.successRate ?? 0}%
            </Descriptions.Item>
          </Descriptions>
        </Card>
      </Card>
    ),
  };

  if (!isValidId) {
    return (
      <div style={{ padding: 48, textAlign: "center" }}>
        <Result
          status="error"
          title="无效的算法ID"
          subTitle="请从列表页选择算法查看详情"
          extra={
            <Button type="primary" onClick={() => navigate("/algorithm")}>
              返回算法列表
            </Button>
          }
        />
      </div>
    );
  }

  return (
    <div className="app-container">
      {/* 页面头部 */}
      <Card style={{ marginBottom: 12 }}>
        <Space align="center" style={{ marginBottom: 8 }}>
          <Button
            type="text"
            icon={<ArrowLeftOutlined />}
            onClick={() => navigate("/algorithm")}
          >
            返回算法列表
          </Button>
        </Space>
        <Space align="center">
          <Title level={3} style={{ margin: 0 }}>
            {algorithm.name}
          </Title>
          <Tag color={statusInfo?.color || "default"}>
            {statusInfo?.label || `未知(${algorithm.status})`}
          </Tag>
          <Tag color={TYPE_COLOR_MAP[algorithm.type] || "blue"}>
            {algorithm.type}
          </Tag>
          {algorithm.version && <Tag>{algorithm.version}</Tag>}
        </Space>
      </Card>

      {/* 算法详情 Tabs */}
      <Tabs
        activeKey="basic"
        items={[basicTab, techTab, opsTab]}
        centered
        size="large"
      />
    </div>
  );
}
