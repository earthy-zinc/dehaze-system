import {
  CloudOutlined,
  DatabaseOutlined,
  ExperimentOutlined,
  ScissorOutlined,
  SplitCellsOutlined,
  SwapOutlined,
  ThunderboltOutlined,
} from "@ant-design/icons";
import { Card, Col, Row, Typography } from "antd";
import React from "react";
import { useNavigate } from "react-router-dom";

const { Title, Paragraph, Text } = Typography;

/** 快捷入口配置 */
const QUICK_ENTRIES = [
  {
    key: "dehaze",
    title: "图像去雾",
    description: "上传雾化图像，选择算法模型进行去雾处理",
    icon: <CloudOutlined />,
    path: "/presentation/dehaze",
    color: "#1890ff",
  },
  {
    key: "segmentation",
    title: "图像分割",
    description: "对图像进行语义分割，识别并标记不同区域",
    icon: <ScissorOutlined />,
    path: "/presentation/segmentation",
    color: "#52c41a",
  },
  {
    key: "overlap",
    title: "重叠对比",
    description: "拖拽分割线对比原图与处理结果",
    icon: <SplitCellsOutlined />,
    path: "/compare/overlap",
    color: "#faad14",
  },
  {
    key: "parallel",
    title: "并排对比",
    description: "多图并排展示，支持放大镜与滤镜",
    icon: <SwapOutlined />,
    path: "/compare/parallel",
    color: "#eb2f96",
  },
  {
    key: "dataset",
    title: "数据集管理",
    description: "浏览和管理系统中的去雾数据集",
    icon: <DatabaseOutlined />,
    path: "/dataset",
    color: "#722ed1",
  },
  {
    key: "algorithm",
    title: "模型管理",
    description: "查看和管理可用的去雾算法模型",
    icon: <ExperimentOutlined />,
    path: "/algorithm",
    color: "#13c2c2",
  },
];

/** 功能特性 */
const FEATURES = [
  {
    title: "高效处理",
    description: "优化的处理流程，支持单张与批量去雾，快速恢复图像清晰度",
    icon: <ThunderboltOutlined />,
  },
  {
    title: "实时反馈",
    description: "5阶段进度监控，清晰展示处理阶段与预估时间",
    icon: <CloudOutlined />,
  },
  {
    title: "参数可调",
    description: "去雾强度、锐化程度等参数自由调节，实现个性化效果",
    icon: <ExperimentOutlined />,
  },
  {
    title: "效果对比",
    description: "重叠对比与并排对比两种模式，放大镜细节查看",
    icon: <SwapOutlined />,
  },
];

const Home: React.FC = () => {
  const navigate = useNavigate();

  return (
    <div style={{ padding: "24px", maxWidth: 1200, margin: "0 auto" }}>
      {/* 系统介绍 */}
      <Card
        style={{
          marginBottom: 24,
          background: "linear-gradient(135deg, #1890ff 0%, #096dd9 100%)",
          border: "none",
        }}
      >
        <div style={{ color: "#fff", textAlign: "center", padding: "20px 0" }}>
          <CloudOutlined style={{ fontSize: 48, marginBottom: 16 }} />
          <Title level={2} style={{ color: "#fff", marginBottom: 8 }}>
            图像去雾系统
          </Title>
          <Paragraph style={{ color: "rgba(255,255,255,0.85)", fontSize: 16 }}>
            基于深度学习的图像去雾处理平台，提供高效的去雾算法展示、效果对比与评估功能。
            支持单张/批量处理、参数调节、多模式对比，帮助您快速恢复雾化图像的清晰度。
          </Paragraph>
        </div>
      </Card>

      {/* 快捷入口 */}
      <Title level={3} style={{ marginBottom: 16 }}>
        快捷入口
      </Title>
      <Row gutter={[16, 16]} style={{ marginBottom: 32 }}>
        {QUICK_ENTRIES.map((entry) => (
          <Col xs={24} sm={12} md={8} key={entry.key}>
            <Card
              hoverable
              onClick={() => navigate(entry.path)}
              style={{ height: "100%", cursor: "pointer" }}
            >
              <div style={{ display: "flex", alignItems: "flex-start", gap: 12 }}>
                <div
                  style={{
                    fontSize: 28,
                    color: entry.color,
                    flexShrink: 0,
                  }}
                >
                  {entry.icon}
                </div>
                <div>
                  <Title level={5} style={{ marginBottom: 4 }}>
                    {entry.title}
                  </Title>
                  <Text type="secondary" style={{ fontSize: 13 }}>
                    {entry.description}
                  </Text>
                </div>
              </div>
            </Card>
          </Col>
        ))}
      </Row>

      {/* 功能特性 */}
      <Title level={3} style={{ marginBottom: 16 }}>
        功能特性
      </Title>
      <Row gutter={[16, 16]}>
        {FEATURES.map((feature) => (
          <Col xs={24} sm={12} md={6} key={feature.title}>
            <Card style={{ height: "100%", textAlign: "center" }}>
              <div style={{ fontSize: 32, color: "#1890ff", marginBottom: 12 }}>
                {feature.icon}
              </div>
              <Title level={5} style={{ marginBottom: 8 }}>
                {feature.title}
              </Title>
              <Text type="secondary" style={{ fontSize: 13 }}>
                {feature.description}
              </Text>
            </Card>
          </Col>
        ))}
      </Row>
    </div>
  );
};

export default Home;
