/**
 * 通用页面布局组件
 *
 * 依据《03-移动端界面设计规范》：页面层级决定导航形态
 * - L1 Tab 根页：底部 TabBar（原生 tabBar，app.config.ts 配置）+ 顶部标题栏（品牌+Tab 标题+搜索）
 * - L2 二级功能页：顶部导航栏（返回+标题），TabBar 隐藏
 * - L3 深度沉浸页：无全局导航（页面内工具栏），由 ImmersiveLayout 负责
 */
import React from "react";
import { View } from "@tarojs/components";
import AppNavbar from "./navbar";
import "./index.less";

export type PageLevel = "L1" | "L2" | "L3";

interface PageLayoutProps {
  /** 子元素 */
  children: React.ReactNode;
  /** 页面层级（决定导航形态） */
  level?: PageLevel;
  /** 页面标题（L1 为 Tab 标题，L2 为页面功能名） */
  title?: string;
  /** 是否为首页（L1 时品牌"去雾"仅在首页显示） */
  isHome?: boolean;
  /** 搜索回调（L1 首页显示搜索入口） */
  onSearch?: () => void;
  /** 右侧操作区（L1 非首页 / L2 按需传入） */
  rightActions?: React.ReactNode;
}

const PageLayout: React.FC<PageLayoutProps> = ({
  children,
  level = "L1",
  title = "",
  isHome = false,
  onSearch,
  rightActions,
}) => {
  return (
    <View className="page-layout">
      {/* 顶部导航栏：L1/L2 显示，L3 沉浸页由页面内工具栏替代 */}
      {level !== "L3" && (
        <AppNavbar
          level={level}
          title={title}
          isHome={isHome}
          onSearch={onSearch}
          rightActions={rightActions}
        />
      )}

      {/* 主内容区（L1 预留原生 tabBar 高度） */}
      <View className={`page-content ${level === "L1" ? "with-tabbar" : ""}`}>
        {children}
      </View>
    </View>
  );
};

export default PageLayout;
export { default as ImmersiveLayout } from "./immersive";
