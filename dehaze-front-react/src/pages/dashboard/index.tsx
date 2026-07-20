import {
  CheckCircleOutlined,
  CloudOutlined,
  DatabaseOutlined,
  ExperimentOutlined,
  FileImageOutlined,
  SyncOutlined,
} from "@ant-design/icons";
import {
  Button,
  Card,
  Col,
  Row,
  Statistic,
  Table,
  Tag,
  Typography,
} from "antd";
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

const { Title } = Typography;

/** 统计卡片数据 */
interface StatCard {
  title: string;
  value: number;
  icon: React.ReactNode;
  color: string;
  suffix?: string;
}

/** 最近任务数据 */
interface RecentTask {
  key: string;
  name: string;
  type: string;
  status: "completed" | "processing" | "failed" | "pending";
  time: string;
}

/** 模拟最近任务数据 */
const RECENT_TASKS: RecentTask[] = [
  {
    key: "1",
    name: "NH-HAZE-118.JPG",
    type: "图像去雾",
    status: "completed",
    time: "2026-07-11 10:30",
  },
  {
    key: "2",
    name: "batch_001.png",
    type: "批量去雾",
    status: "processing",
    time: "2026-07-11 10:25",
  },
  {
    key: "3",
    name: "Dense-Haze-01.png",
    type: "图像去雾",
    status: "completed",
    time: "2026-07-11 09:45",
  },
  {
    key: "4",
    name: "segment_05.jpg",
    type: "图像分割",
    status: "failed",
    time: "2026-07-11 09:20",
  },
  {
    key: "5",
    name: "OTS_3000.jpg",
    type: "图像去雾",
    status: "pending",
    time: "2026-07-11 08:50",
  },
];

/** 任务状态标签映射 */
const STATUS_TAG_MAP: Record<
  RecentTask["status"],
  { color: string; text: string; icon: React.ReactNode }
> = {
  completed: {
    color: "success",
    text: "已完成",
    icon: <CheckCircleOutlined />,
  },
  processing: {
    color: "processing",
    text: "处理中",
    icon: <SyncOutlined spin />,
  },
  failed: { color: "error", text: "失败", icon: <CheckCircleOutlined /> },
  pending: { color: "default", text: "等待中", icon: <SyncOutlined /> },
};

const Dashboard: React.FC = () => {
  const navigate = useNavigate();
  const [stats] = useState<StatCard[]>([
    {
      title: "已处理图片",
      value: 1286,
      icon: <FileImageOutlined />,
      color: "#3B82F6",
      suffix: "张",
    },
    {
      title: "可用模型",
      value: 12,
      icon: <ExperimentOutlined />,
      color: "#52c41a",
      suffix: "个",
    },
    {
      title: "数据集",
      value: 8,
      icon: <DatabaseOutlined />,
      color: "#faad14",
      suffix: "个",
    },
    {
      title: "今日任务",
      value: 36,
      icon: <CloudOutlined />,
      color: "#eb2f96",
      suffix: "个",
    },
  ]);

  // 表格列定义
  const columns = [
    {
      title: "文件名",
      dataIndex: "name",
      key: "name",
    },
    {
      title: "任务类型",
      dataIndex: "type",
      key: "type",
    },
    {
      title: "状态",
      dataIndex: "status",
      key: "status",
      render: (status: RecentTask["status"]) => {
        const tag = STATUS_TAG_MAP[status];
        return (
          <Tag color={tag.color} icon={tag.icon}>
            {tag.text}
          </Tag>
        );
      },
    },
    {
      title: "时间",
      dataIndex: "time",
      key: "time",
    },
  ];

  return (
    <div style={{ padding: "24px", maxWidth: 1200, margin: "0 auto" }}>
      <Title level={3} style={{ marginBottom: 24 }}>
        数据看板
      </Title>

      {/* 统计卡片 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        {stats.map((stat) => (
          <Col xs={24} sm={12} md={6} key={stat.title}>
            <Card>
              <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
                <div
                  style={{
                    fontSize: 36,
                    color: stat.color,
                  }}
                >
                  {stat.icon}
                </div>
                <Statistic
                  title={stat.title}
                  value={stat.value}
                  suffix={stat.suffix}
                />
              </div>
            </Card>
          </Col>
        ))}
      </Row>

      {/* 快捷操作 */}
      <Card title="快捷操作" style={{ marginBottom: 24 }}>
        <Row gutter={[16, 16]}>
          <Col xs={24} sm={12} md={6}>
            <Button
              block
              type="primary"
              icon={<CloudOutlined />}
              onClick={() => navigate("/presentation/dehaze")}
            >
              开始去雾
            </Button>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Button
              block
              icon={<ExperimentOutlined />}
              onClick={() => navigate("/algorithm")}
            >
              管理模型
            </Button>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Button
              block
              icon={<DatabaseOutlined />}
              onClick={() => navigate("/dataset")}
            >
              浏览数据集
            </Button>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Button
              block
              icon={<FileImageOutlined />}
              onClick={() => navigate("/compare/parallel")}
            >
              效果对比
            </Button>
          </Col>
        </Row>
      </Card>

      {/* 最近任务 */}
      <Card title="最近任务">
        <Table
          dataSource={RECENT_TASKS}
          columns={columns}
          pagination={{ pageSize: 5 }}
          size="middle"
        />
      </Card>
    </div>
  );
};

export default Dashboard;
