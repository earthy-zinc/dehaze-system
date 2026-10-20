import React from "react";
import { View, Text, ScrollView } from "@tarojs/components";
import Taro from "@tarojs/taro";
import PageLayout from "@/layout";
import "./index.less";

const SettingsPage: React.FC = () => {
  const handleClearCache = () => {
    Taro.showToast({ title: "缓存已清理", icon: "success" });
  };

  const handleAbout = () => {
    Taro.navigateTo({ url: "/pages/personal/about/index" });
  };

  return (
    <PageLayout level="L2" title="系统设置">
      <View className="personal-settings-page">
        <ScrollView scrollY className="settings-scroll">
          <View className="settings-group">
            <Text className="settings-group-title">通用</Text>
            <View className="settings-card">
              <View className="settings-item">
                <View className="settings-item-left">
                  <Text className="settings-icon">🌙</Text>
                  <Text className="settings-title">暗色模式</Text>
                </View>
                <Text className="settings-value">跟随系统</Text>
              </View>
              <View className="settings-divider" />
              <View className="settings-item" onClick={handleClearCache}>
                <View className="settings-item-left">
                  <Text className="settings-icon">🗑️</Text>
                  <Text className="settings-title">清理缓存</Text>
                </View>
                <Text className="settings-arrow">›</Text>
              </View>
            </View>
          </View>

          <View className="settings-group">
            <Text className="settings-group-title">其他</Text>
            <View className="settings-card">
              <View className="settings-item" onClick={handleAbout}>
                <View className="settings-item-left">
                  <Text className="settings-icon">ℹ️</Text>
                  <Text className="settings-title">关于我们</Text>
                </View>
                <Text className="settings-arrow">›</Text>
              </View>
            </View>
          </View>
        </ScrollView>
      </View>
    </PageLayout>
  );
};

export default SettingsPage;
