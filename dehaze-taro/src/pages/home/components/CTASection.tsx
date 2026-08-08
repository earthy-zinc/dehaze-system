import React from "react";
import { View, Text, Button } from "@tarojs/components";
import Taro from "@tarojs/taro";
import { Arrow } from "@taroify/icons";

import "./CTASection.less";

const CTASection: React.FC = () => {
  const handleStartClick = () => {
    Taro.switchTab({ url: "/pages/dehaze/index" });
  };

  return (
    <View className="final-cta-section">
      <Text className="cta-title">准备好体验专业级图像去雾了吗？</Text>
      <Text className="cta-subtitle">立即开始，让您的图像重获清晰</Text>
      <Button className="cta-large-btn" onClick={handleStartClick}>
        开始去雾
        <Arrow className="cta-icon" size="16" color="#ffffff" />
      </Button>
    </View>
  );
};

export default CTASection;
