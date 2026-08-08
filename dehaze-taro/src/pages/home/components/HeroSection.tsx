import React from "react";
import { View, Text, Button } from "@tarojs/components";
import Taro from "@tarojs/taro";
import { Arrow } from "@taroify/icons";

import "./HeroSection.less";

const HeroSection: React.FC = () => {
  const handleStartClick = () => {
    Taro.switchTab({ url: "/pages/dehaze/index" });
  };

  const handleDatasetClick = () => {
    Taro.navigateTo({ url: "/pages/dataset/index" });
  };

  return (
    <View className="hero-section">
      <View className="hero-content">
        <Text className="hero-title">图像去雾</Text>
        <Text className="hero-subtitle">专业级图像处理系统</Text>
        <View className="hero-description">
          <Text>采用先进的深度学习算法，一键还原清晰视界</Text>
          <Text>从图像输入到效果评估的完整闭环体验</Text>
        </View>
        <View className="hero-cta">
          <Button className="cta-primary" onClick={handleStartClick}>
            开始去雾
            <Arrow className="cta-icon" size="14" color="#ffffff" />
          </Button>
          <Button className="cta-secondary" onClick={handleDatasetClick}>
            浏览数据集
          </Button>
        </View>
      </View>
    </View>
  );
};

export default HeroSection;
