/**
 * 布局相关 Hook
 * 提供响应式布局判断和侧边栏控制
 */
import { useState, useEffect, useCallback } from "react";
import Taro from "@tarojs/taro";

/** 布局断点配置 */
const BREAKPOINTS = {
  /** 平板断点 */
  tablet: 768,
  /** 桌面断点 */
  desktop: 1024,
  /** 大屏断点 */
  largeDesktop: 1440,
};

interface LayoutState {
  /** 屏幕宽度 */
  screenWidth: number;
  /** 屏幕高度 */
  screenHeight: number;
  /** 状态栏高度 */
  statusBarHeight: number;
  /** 是否为宽屏设备 */
  isWideScreen: boolean;
  /** 是否为平板设备 */
  isTablet: boolean;
  /** 是否为桌面设备 */
  isDesktop: boolean;
  /** 是否为移动端 */
  isMobile: boolean;
  /** 是否为横屏模式 */
  isLandscape: boolean;
}

/**
 * 布局 Hook
 * 提供响应式布局状态
 */
export function useLayout(): LayoutState {
  const [state, setState] = useState<LayoutState>({
    screenWidth: 375,
    screenHeight: 667,
    statusBarHeight: 0,
    isWideScreen: false,
    isTablet: false,
    isDesktop: false,
    isMobile: true,
    isLandscape: false,
  });

  useEffect(() => {
    const updateScreenSize = () => {
      try {
        const sysInfo = Taro.getSystemInfoSync();
        const width = sysInfo.windowWidth || 375;
        const height = sysInfo.windowHeight || 667;

        setState({
          screenWidth: width,
          screenHeight: height,
          statusBarHeight: sysInfo.statusBarHeight || 0,
          isWideScreen: width >= BREAKPOINTS.tablet,
          isTablet: width >= BREAKPOINTS.tablet && width < BREAKPOINTS.desktop,
          isDesktop: width >= BREAKPOINTS.desktop,
          isMobile: width < BREAKPOINTS.tablet,
          isLandscape: width > height,
        });
      } catch (error) {
        void error;
      }
    };

    updateScreenSize();

    // H5 环境监听窗口变化
    if (process.env.TARO_ENV === "h5") {
      window.addEventListener("resize", updateScreenSize);
      return () => window.removeEventListener("resize", updateScreenSize);
    }
  }, []);

  return state;
}

/**
 * 侧边栏控制 Hook
 */
export function useSidebar() {
  const [visible, setVisible] = useState(false);

  const open = useCallback(() => setVisible(true), []);
  const close = useCallback(() => setVisible(false), []);
  const toggle = useCallback(() => setVisible((v) => !v), []);

  return {
    visible,
    open,
    close,
    toggle,
  };
}

/**
 * 获取当前页面路由
 */
export function useCurrentRoute(): string {
  const [route, setRoute] = useState("/pages/home/index");

  useEffect(() => {
    const pages = Taro.getCurrentPages();
    if (pages.length > 0) {
      const currentPage = pages[pages.length - 1];
      setRoute("/" + (currentPage.route || "pages/home/index"));
    }
  }, []);

  return route;
}

/**
 * 获取状态栏高度
 */
export function useStatusBarHeight(): number {
  const [height, setHeight] = useState(0);

  useEffect(() => {
    try {
      const sysInfo = Taro.getSystemInfoSync();
      setHeight(sysInfo.statusBarHeight || 0);
    } catch (error) {
      void error;
    }
  }, []);

  return height;
}
