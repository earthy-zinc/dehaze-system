/**
 * 对比页面通用顶部导航栏
 */
import React from "react";
import { Icon, View, Text } from "@/platform/ui";
import Taro from "@tarojs/taro";

import "./index.less";

interface CompareNavbarProps {
  title: string;
}

const CompareNavbar: React.FC<CompareNavbarProps> = ({ title }) => {
  return (
    <View className="navbar">
      <View className="nav-back" onClick={() => Taro.navigateBack()}>
        <Icon name="arrow-left" size="20" color="#333" />
      </View>
      <Text className="nav-title">{title}</Text>
    </View>
  );
};

export default CompareNavbar;
