import React from "react";
import { View, Text } from "@tarojs/components";
import Taro from "@tarojs/taro";
import {
  BarChartOutlined,
  Exchange,
  Search,
  BrushOutlined,
  ChartTrendingOutlined,
  GoodJobOutlined,
} from "@taroify/icons";

import "./ToolsGrid.less";

interface ToolCardProps {
  icon: React.ReactNode;
  title: string;
  description: string;
  target: string;
}

const ToolCard: React.FC<ToolCardProps> = ({
  icon,
  title,
  description,
  target,
}) => {
  const handleClick = () => {
    Taro.navigateTo({ url: `/pages/${target}/index` });
  };

  return (
    <View className="tool-card" onClick={handleClick}>
      <View className="tool-icon-wrapper">{icon}</View>
      <Text className="tool-title">{title}</Text>
      <View className="tool-desc">
        <Text>{description}</Text>
      </View>
    </View>
  );
};

const ToolsGrid: React.FC = () => {
  const tools = [
    {
      icon: <BarChartOutlined size="24" color="#3b82f6" />,
      title: "并排对比",
      description: "多图并排展示，支持2-4张图片同屏对比",
      target: "side-by-side",
    },
    {
      icon: <Exchange size="24" color="#3b82f6" />,
      title: "重叠对比",
      description: "拖动分割线实时对比，支持横向和纵向模式",
      target: "overlay",
    },
    {
      icon: <Search size="24" color="#3b82f6" />,
      title: "放大镜",
      description: "局部细节放大查看，精确对比图像质量",
      target: "magnifier",
    },
    {
      icon: <BrushOutlined size="24" color="#3b82f6" />,
      title: "滤镜调节",
      description: "实时调节亮度、对比度、饱和度等参数",
      target: "filter",
    },
    {
      icon: <ChartTrendingOutlined size="24" color="#3b82f6" />,
      title: "指标评估",
      description: "SSIM、PSNR等专业指标定量分析",
      target: "metrics",
    },
    {
      icon: <GoodJobOutlined size="24" color="#3b82f6" />,
      title: "数据集管理",
      description: "浏览和管理多个专业去雾数据集",
      target: "dataset",
    },
  ];

  return (
    <View className="tools-section">
      <View className="tools-grid">
        {tools.map((tool) => (
          <ToolCard
            key={tool.target}
            icon={tool.icon}
            title={tool.title}
            description={tool.description}
            target={tool.target}
          />
        ))}
      </View>
    </View>
  );
};

export default ToolsGrid;
