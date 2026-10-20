import React, { useState, useEffect } from "react";
import { View, Text } from "@tarojs/components";
import {
  Fire,
  PhoneOutlined,
  BulbOutlined,
  ChartTrendingOutlined,
} from "@taroify/icons";
import { AiModelAPI } from "clarity-sdk-js";

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

const TechSpecs: React.FC = () => {
  const [modelCount, setModelCount] = useState<number | null>(null);

  useEffect(() => {
    const fetchModelCount = async () => {
      try {
        const res = await AiModelAPI.listModels({
          pageNum: 1,
          pageSize: 1,
          modelType: "image_edit",
        });
        setModelCount(res.total ?? null);
      } catch {
        setModelCount(null);
      }
    };
    fetchModelCount();
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
      title: "智能模型",
      value: modelCount === null ? "-" : `${modelCount}`,
      description: "支持多种先进去雾模型",
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
        {specs.map((spec) => (
          <SpecCard
            key={spec.title}
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
