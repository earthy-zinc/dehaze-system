/**
 * L3 沉浸页统一布局组件
 *
 * 依据《03-移动端界面设计规范》4.4：
 * - 顶部：深色半透明导航栏（返回 + 标题 + 右侧操作插槽）
 * - 底部：深色工具栏（操作按钮插槽）
 * - 内容区全屏沉浸，无全局导航/TabBar
 */
import React from "react";
import { View, Text } from "@tarojs/components";
import Taro from "@tarojs/taro";
import { ArrowLeft } from "@taroify/icons";
import "./index.less";

interface ImmersiveLayoutProps {
  /** 导航栏标题 */
  title: string;
  /** 页面内容（全屏） */
  children: React.ReactNode;
  /** 右侧操作区（可选，如"编辑/分享"） */
  rightActions?: React.ReactNode;
  /** 底部工具栏（可选，如模式切换+操作按钮） */
  toolbar?: React.ReactNode;
}

const ImmersiveLayout: React.FC<ImmersiveLayoutProps> = ({
  title,
  children,
  rightActions,
  toolbar,
}) => {
  const goBack = () => {
    Taro.navigateBack();
  };

  return (
    <View className="immersive-layout">
      {/* 顶部深色半透明导航栏 */}
      <View className="immersive-navbar">
        <View className="navbar-back" onClick={goBack}>
          <ArrowLeft size="20" color="#fff" />
        </View>
        <Text className="navbar-title">{title}</Text>
        <View className="navbar-actions">{rightActions}</View>
      </View>

      {/* 全屏内容区 */}
      <View className="immersive-body">{children}</View>

      {/* 底部工具栏 */}
      {toolbar && <View className="immersive-toolbar">{toolbar}</View>}
    </View>
  );
};

export default ImmersiveLayout;
