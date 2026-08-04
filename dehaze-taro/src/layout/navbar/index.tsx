/**
 * 顶部导航栏组件
 *
 * L1（Tab 根页）：品牌标识 + Tab 标题 + 搜索入口
 * L2（二级功能页）：返回按钮 + 居中页面标题
 */
import React from "react";
import { View, Text } from "@tarojs/components";
import Taro from "@tarojs/taro";
import { Search, ArrowLeft } from "@taroify/icons";
import { useStatusBarHeight } from "@/hooks/useLayout";
import "./index.less";

interface AppNavbarProps {
  /** 导航形态：L1 Tab 根页 / L2 二级功能页 */
  level?: "L1" | "L2";
  /** 页面标题：L1 为 Tab 标题，L2 为页面功能名 */
  title?: string;
  /** 搜索回调（L1 显示） */
  onSearch?: () => void;
}

const AppNavbar: React.FC<AppNavbarProps> = ({
  level = "L1",
  title = "",
  onSearch,
}) => {
  const statusBarHeight = useStatusBarHeight();

  /** 返回上一页；无历史则回首页 */
  const goBack = () => {
    const pages = Taro.getCurrentPages();
    if (pages.length > 1) {
      Taro.navigateBack();
    } else {
      Taro.switchTab({ url: "/pages/home/index" });
    }
  };

  /** 跳转首页（Tab 根页面，使用 switchTab） */
  const goHome = () => {
    Taro.switchTab({ url: "/pages/home/index" });
  };

  /** 搜索按钮点击 */
  const handleSearch = () => {
    if (onSearch) {
      onSearch();
    } else {
      Taro.showToast({ title: "搜索功能开发中", icon: "none" });
    }
  };

  return (
    <View className="app-navbar">
      {/* 状态栏占位 */}
      <View className="status-bar" style={{ height: `${statusBarHeight}px` }} />

      {/* 导航栏内容 */}
      <View className="navbar-content">
        {/* L2：返回 */}
        {level === "L2" ? (
          <View className="navbar-back" onClick={goBack}>
            <ArrowLeft size="18" color="#374151" />
          </View>
        ) : (
          /* L1：品牌标识（点击回首页）+ Tab 标题 */
          <View className="navbar-brand" onClick={goHome}>
            <View className="logo-wrapper">
              <Text className="logo-text">去雾</Text>
            </View>
            <Text className="app-title">{title}</Text>
          </View>
        )}

        {/* L2 居中页面标题 */}
        {level === "L2" && title && (
          <Text className="navbar-title">{title}</Text>
        )}

        {/* 右侧操作区：L1 显示全局搜索入口 */}
        <View className="navbar-actions">
          {level === "L1" && (
            <View className="action-btn" onClick={handleSearch}>
              <Search size="18" color="#374151" />
            </View>
          )}
        </View>
      </View>
    </View>
  );
};

export default AppNavbar;
