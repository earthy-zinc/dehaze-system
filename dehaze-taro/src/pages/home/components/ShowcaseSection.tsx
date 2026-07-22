import React from "react";
import { View, Text } from "@tarojs/components";

import { apiConfig } from "@/config/api";
import ComparisonItem from "./ComparisonItem";
import "./ShowcaseSection.less";

const ShowcaseSection: React.FC = () => {
  // 使用 nginx-dataset 提供的 NH-HAZE-2023 样张
  const showcaseImageUrl = `${apiConfig.dataset}/NH-HAZE-2023/hazy/01_hazy.png`;

  return (
    <View className="showcase-section">
      <View className="showcase-header">
        <Text className="section-title">一键去雾，效果显著</Text>
        <Text className="section-subtitle">
          智能算法自动识别雾霾程度，精准还原图像细节
        </Text>
      </View>
      <View className="comparison-showcase">
        <ComparisonItem imageUrl={showcaseImageUrl} />
      </View>
    </View>
  );
};

export default ShowcaseSection;
