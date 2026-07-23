/**
 * 侧边菜单组件
 */
import React from "react";
import { View, Text, ScrollView } from "@tarojs/components";
import Taro from "@tarojs/taro";
import { Popup } from "@taroify/core";
import { Cross, EyeOutlined } from "@taroify/icons";
import { homeItem, menuSections, type MenuItem } from "@/config/menu";
import "./index.less";

interface AppSidebarProps {
  /** 侧边栏是否可见 */
  visible: boolean;
  /** 当前路由 */
  currentRoute?: string;
  /** 关闭回调 */
  onClose: () => void;
  /** 导航回调 */
  onNavigate?: (route: string) => void;
}

const AppSidebar: React.FC<AppSidebarProps> = ({
  visible,
  currentRoute = "/pages/home/index",
  onClose,
  onNavigate,
}) => {
  /** 判断是否为当前激活路由 */
  const isActive = (route: string) => currentRoute === route;

  /** 导航到指定路由 */
  const navigateTo = (route: string) => {
    onNavigate?.(route);
    onClose();

    // 使用 reLaunch 统一跳转（项目使用自定义 tabbar 组件，未配置原生 tabBar）
    Taro.reLaunch({
      url: route,
      fail: () => {
        Taro.showToast({ title: "页面开发中", icon: "none" });
      },
    });
  };

  /** 渲染菜单项 */
  const renderMenuItem = (item: MenuItem) => (
    <View
      key={item.route}
      className={`menu-item ${isActive(item.route) ? "active" : ""}`}
      onClick={() => navigateTo(item.route)}
    >
      <Text className="menu-title">{item.title}</Text>
    </View>
  );

  return (
    <Popup
      open={visible}
      placement="right"
      style={{ width: "70vw", maxWidth: "280px" }}
      onClose={onClose}
    >
      <View className="sidebar-container">
        {/* 头部 */}
        <View className="sidebar-header">
          <View className="header-content">
            <View className="logo-wrapper">
              <EyeOutlined size="18" color="#ffffff" />
            </View>
            <View className="header-text">
              <Text className="app-name">图像去雾系统</Text>
              <Text className="app-desc">功能菜单</Text>
            </View>
          </View>
          <View className="close-btn" onClick={onClose}>
            <Cross size="16" color="#fff" />
          </View>
        </View>

        {/* 菜单内容 */}
        <ScrollView className="sidebar-content" scrollY>
          {/* 首页 */}
          {renderMenuItem(homeItem)}

          <View className="menu-divider" />

          {/* 分组菜单 */}
          {menuSections.map((section) => (
            <View key={section.title} className="menu-section">
              <View className="section-header">
                <Text className="section-title">{section.title}</Text>
              </View>
              {section.items.map(renderMenuItem)}
            </View>
          ))}
        </ScrollView>
      </View>
    </Popup>
  );
};

export default AppSidebar;
