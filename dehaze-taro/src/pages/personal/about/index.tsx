import React from "react";
import { View, Text, ScrollView } from "@tarojs/components";
import PageLayout from "@/layout";
import "./index.less";

const AboutPage: React.FC = () => {
  return (
    <PageLayout level="L2" title="关于我们">
      <View className="personal-about-page">
        <ScrollView scrollY className="about-scroll">
          <View className="about-header">
            <Text className="about-logo">🌫️</Text>
            <Text className="about-app-name">图像去雾系统</Text>
            <Text className="about-version">v1.0.0</Text>
          </View>

          <View className="about-card">
            <Text className="about-card-title">产品简介</Text>
            <Text className="about-card-text">
              Dehaze
              是一套基于深度学习的图像去雾处理系统，支持多种主流去雾算法，提供高效的图像处理体验。用户可通过简单的操作完成高质量的去雾处理，并支持效果对比与批量处理。
            </Text>
          </View>

          <View className="about-card">
            <Text className="about-card-title">技术栈</Text>
            <View className="about-tech-list">
              <Text className="about-tech-item">深度学习算法模型</Text>
              <Text className="about-tech-item">
                Java / Python / Go 多端后端
              </Text>
              <Text className="about-tech-item">
                React / Vue / Flutter 多端前端
              </Text>
              <Text className="about-tech-item">Taro 跨端小程序</Text>
            </View>
          </View>

          <View className="about-card">
            <Text className="about-card-title">法律信息</Text>
            <View className="about-legal-list">
              <View className="about-legal-item">
                <Text className="about-legal-title">用户协议</Text>
                <Text className="about-legal-arrow">›</Text>
              </View>
              <View className="about-legal-divider" />
              <View className="about-legal-item">
                <Text className="about-legal-title">隐私政策</Text>
                <Text className="about-legal-arrow">›</Text>
              </View>
            </View>
          </View>

          <View className="about-footer">
            <Text className="about-footer-text">
              Copyright © {new Date().getFullYear()} Dehaze Team. All rights
              reserved.
            </Text>
          </View>
        </ScrollView>
      </View>
    </PageLayout>
  );
};

export default AboutPage;
