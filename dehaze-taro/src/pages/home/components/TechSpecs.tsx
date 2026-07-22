import React, { useState, useEffect } from "react";
import { View, Text } from "@tarojs/components";
import {
  Fire,
  PhoneOutlined,
  BulbOutlined,
  ChartTrendingOutlined,
} from "@taroify/icons";
import { AlgorithmAPI } from "dehaze-sdk-js";
import type { Algorithm } from "dehaze-sdk-js";

import "./TechSpecs.less";

interface SpecCardProps {
  icon: React.ReactNode;
  title: string;
  value: string;
  description: string;
}

const SpecCard: React.FC<SpecCardProps> = ({
  icon,
  title,
  value,
  description,
}) => {
  return (
    <View className="spec-card">
      <View className="spec-icon">{icon}</View>
      <Text className="spec-title">{title}</Text>
      <Text className="spec-value">{value}</Text>
      <View className="spec-desc">
        <Text>{description}</Text>
      </View>
    </View>
  );
};

/** 递归统计算法树节点总数 */
function countAlgorithmNodes(nodes: Algorithm[]): number {
  return nodes.reduce(
    (sum, n) => sum + 1 + (n.children ? countAlgorithmNodes(n.children) : 0),
    0
  );
}

const TechSpecs: React.FC = () => {
  const [algorithmCount, setAlgorithmCount] = useState<number | null>(null);

  useEffect(() => {
    const fetchAlgorithmCount = async () => {
      try {
        const tree = await AlgorithmAPI.getList();
        setAlgorithmCount(countAlgorithmNodes(tree || []));
      } catch {
        setAlgorithmCount(null);
      }
    };
    fetchAlgorithmCount();
  }, []);

  const specs = [
    {
      icon: <Fire size="28" color="#ffffff" />,
      title: "高性能",
      value: "实时",
      description: "后端 GPU 加速去雾处理",
    },
    {
      icon: <PhoneOutlined size="28" color="#ffffff" />,
      title: "全平台",
      value: "H5·小程序",
      description: "适配手机、平板、桌面浏览器",
    },
    {
      icon: <BulbOutlined size="28" color="#ffffff" />,
      title: "智能算法",
      value: algorithmCount === null ? "-" : `${algorithmCount}`,
      description: "支持多种先进去雾算法",
    },
    {
      icon: <ChartTrendingOutlined size="28" color="#ffffff" />,
      title: "专业评估",
      value: "多维",
      description: "PSNR / SSIM 等定量指标",
    },
  ];

  return (
    <View className="tech-specs-section">
      <View className="specs-grid">
        {specs.map((spec, index) => (
          <SpecCard
            key={index}
            icon={spec.icon}
            title={spec.title}
            value={spec.value}
            description={spec.description}
          />
        ))}
      </View>
    </View>
  );
};

export default TechSpecs;
