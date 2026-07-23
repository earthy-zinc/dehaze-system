import {
  CloudOutlined,
  DatabaseOutlined,
  ExperimentOutlined,
  EyeOutlined,
  ScissorOutlined,
  SplitCellsOutlined,
  SwapOutlined,
  ThunderboltOutlined,
  TrophyOutlined,
} from "@ant-design/icons";
import { Avatar, Card, Col, Row } from "antd";
import React, { useEffect, useMemo, useState } from "react";
import { useSelector } from "react-redux";
import { useNavigate } from "react-router-dom";
import { RootState } from "@/store";
import "./index.scss";

/** 快捷入口配置：图标色块使用品牌渐变与功能色（参考 §2.4 渐变色系统） */
const QUICK_ENTRIES = [
  {
    key: "dehaze",
    title: "图像去雾",
    description: "上传雾化图像，选择算法模型进行去雾处理",
    icon: <CloudOutlined />,
    path: "/presentation/dehaze",
    gradient: "linear-gradient(135deg, #3b82f6 0%, #6366f1 100%)",
  },
  {
    key: "segmentation",
    title: "图像分割",
    description: "对图像进行语义分割，识别并标记不同区域",
    icon: <ScissorOutlined />,
    path: "/presentation/segmentation",
    gradient: "linear-gradient(135deg, #4caf50 0%, #388e3c 100%)",
  },
  {
    key: "overlap",
    title: "重叠对比",
    description: "拖拽分割线对比原图与处理结果",
    icon: <SplitCellsOutlined />,
    path: "/compare/overlap",
    gradient: "linear-gradient(135deg, #ff9800 0%, #f57c00 100%)",
  },
  {
    key: "parallel",
    title: "并排对比",
    description: "多图并排展示，支持放大镜与滤镜",
    icon: <SwapOutlined />,
    path: "/compare/parallel",
    gradient: "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
  },
  {
    key: "dataset",
    title: "数据集管理",
    description: "浏览和管理系统中的去雾数据集",
    icon: <DatabaseOutlined />,
    path: "/dataset",
    gradient: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
  },
  {
    key: "algorithm",
    title: "模型管理",
    description: "查看和管理可用的去雾算法模型",
    icon: <ExperimentOutlined />,
    path: "/algorithm",
    gradient: "linear-gradient(135deg, #13c2c2 0%, #08979c 100%)",
  },
] as const;

/** 功能特性 */
const FEATURES = [
  {
    title: "高效处理",
    description: "优化的处理流程，支持单张与批量去雾，快速恢复图像清晰度",
    icon: <ThunderboltOutlined />,
    gradient: "linear-gradient(135deg, #3b82f6 0%, #6366f1 100%)",
  },
  {
    title: "实时反馈",
    description: "5 阶段进度监控，清晰展示处理阶段与预估时间",
    icon: <CloudOutlined />,
    gradient: "linear-gradient(135deg, #4caf50 0%, #388e3c 100%)",
  },
  {
    title: "参数可调",
    description: "去雾强度、锐化程度等参数自由调节，实现个性化效果",
    icon: <ExperimentOutlined />,
    gradient: "linear-gradient(135deg, #ff9800 0%, #f57c00 100%)",
  },
  {
    title: "效果对比",
    description: "重叠对比与并排对比两种模式，放大镜细节查看",
    icon: <EyeOutlined />,
    gradient: "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
  },
] as const;

/** 根据小时数返回问候语 */
function getGreeting(hours: number): string {
  if (hours >= 6 && hours < 8) return "晨起披衣出草堂，轩窗已自喜微凉 🌅";
  if (hours >= 8 && hours < 12) return "上午好";
  if (hours >= 12 && hours < 14) return "中午好";
  if (hours >= 14 && hours < 18) return "下午好";
  if (hours >= 18 && hours < 24) return "晚上好";
  return "夜深了，注意休息哦 🌙";
}

/** 格式化日期 */
function formatDate(date: Date): string {
  const weekdays = ["周日", "周一", "周二", "周三", "周四", "周五", "周六"];
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, "0");
  const d = String(date.getDate()).padStart(2, "0");
  return `${y}-${m}-${d} ${weekdays[date.getDay()]}`;
}

const Home: React.FC = () => {
  const navigate = useNavigate();
  const userStore = useSelector((state: RootState) => state.user);
  const permissionStore = useSelector((state: RootState) => state.permission);

  const [now, setNow] = useState(() => new Date());

  useEffect(() => {
    // 每分钟刷新一次时间，保证问候语及时更新
    const timer = setInterval(() => setNow(new Date()), 60_000);
    return () => clearInterval(timer);
  }, []);

  const greeting = useMemo(() => getGreeting(now.getHours()), [now]);
  const dateStr = useMemo(() => formatDate(now), [now]);

  const nickname = userStore.user.nickname || userStore.user.username || "用户";
  const roles = userStore.user.roles || [];
  const roleLabel = roles.includes("ROOT")
    ? "超级管理员"
    : roles.includes("ROLE_ADMIN")
      ? "系统管理员"
      : roles[0] || "普通用户";

  // 取首字符作为头像占位
  const avatarChar = nickname.charAt(0).toUpperCase();
  const avatarUrl = userStore.user.avatar;

  // 顶部统计：菜单数 + 权限数 + 角色 + 系统版本
  const menuCount = permissionStore.routes?.length ?? 0;
  const permCount = userStore.user.permissions?.length ?? 0;

  const stats = [
    { label: "可用菜单", value: menuCount },
    { label: "功能权限", value: permCount },
    { label: "担任角色", value: roles.length },
  ];

  return (
    <div className="home-container">
      {/* 欢迎卡 */}
      <Card className="welcome-card" variant="borderless">
        <div className="welcome-content">
          <div className="welcome-left">
            <div className="welcome-avatar">
              {avatarUrl ? (
                <Avatar src={avatarUrl} size={58} />
              ) : (
                <Avatar size={58}>{avatarChar}</Avatar>
              )}
            </div>
            <div className="welcome-text">
              <h2 className="greeting">
                {greeting}
                {greeting.length <= 4 ? `，${nickname}！` : ""}
              </h2>
              <p className="subtitle">
                欢迎使用图像去雾系统，今天是 {dateStr}
              </p>
              <span className="role-tag">{roleLabel}</span>
            </div>
          </div>
          <div className="welcome-right">
            {stats.map((s) => (
              <div className="stat-item" key={s.label}>
                <div className="stat-value">{s.value}</div>
                <div className="stat-label">{s.label}</div>
              </div>
            ))}
          </div>
        </div>
      </Card>

      {/* 快捷入口 */}
      <h3 className="section-title">快捷入口</h3>
      <Row gutter={[16, 16]}>
        {QUICK_ENTRIES.map((entry) => (
          <Col xs={24} sm={12} lg={8} key={entry.key}>
            <Card
              className="quick-entry-card"
              variant="borderless"
              onClick={() => navigate(entry.path)}
            >
              <div className="entry-content">
                <div
                  className="entry-icon"
                  style={{ background: entry.gradient }}
                >
                  {entry.icon}
                </div>
                <div className="entry-text">
                  <h4 className="entry-title">{entry.title}</h4>
                  <p className="entry-desc">{entry.description}</p>
                </div>
              </div>
            </Card>
          </Col>
        ))}
      </Row>

      {/* 功能特性 */}
      <h3 className="section-title">
        <TrophyOutlined style={{ marginRight: 8, color: "#3b82f6" }} />
        功能特性
      </h3>
      <Row gutter={[16, 16]}>
        {FEATURES.map((feature) => (
          <Col xs={24} sm={12} lg={6} key={feature.title}>
            <Card className="feature-card" variant="borderless">
              <div
                className="feature-icon"
                style={{ background: feature.gradient }}
              >
                {feature.icon}
              </div>
              <h4 className="feature-title">{feature.title}</h4>
              <p className="feature-desc">{feature.description}</p>
            </Card>
          </Col>
        ))}
      </Row>
    </div>
  );
};

export default Home;
