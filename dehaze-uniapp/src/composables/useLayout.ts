/**
 * 布局相关 Composable
 * 提供响应式布局判断和侧边栏控制
 */
import { ref, computed, onMounted, onUnmounted } from "vue";

/** 布局断点配置 */
const BREAKPOINTS = {
  /** 平板断点 */
  tablet: 768,
  /** 桌面断点 */
  desktop: 1024,
  /** 大屏断点 */
  largeDesktop: 1440,
};

/**
 * 布局 Hook
 * 提供响应式布局状态和侧边栏控制
 */
export function useLayout() {
  /** 屏幕宽度 */
  const screenWidth = ref(375);
  /** 屏幕高度 */
  const screenHeight = ref(667);
  /** 侧边栏可见状态 */
  const sidebarVisible = ref(false);

  // ==================== 响应式断点判断 ====================

  /** 是否为宽屏设备（平板/桌面，≥768px） */
  const isWideScreen = computed(() => screenWidth.value >= BREAKPOINTS.tablet);

  /** 是否为平板设备（768px - 1023px） */
  const isTablet = computed(
    () =>
      screenWidth.value >= BREAKPOINTS.tablet &&
      screenWidth.value < BREAKPOINTS.desktop
  );

  /** 是否为桌面设备（≥1024px） */
  const isDesktop = computed(() => screenWidth.value >= BREAKPOINTS.desktop);

  /** 是否为大屏桌面（≥1440px） */
  const isLargeDesktop = computed(
    () => screenWidth.value >= BREAKPOINTS.largeDesktop
  );

  /** 是否为移动端（<768px） */
  const isMobile = computed(() => screenWidth.value < BREAKPOINTS.tablet);

  /** 是否为横屏模式 */
  const isLandscape = computed(() => screenWidth.value > screenHeight.value);

  // ==================== 屏幕尺寸更新 ====================

  /** 更新屏幕尺寸 */
  const updateScreenSize = () => {
    try {
      const sysInfo = uni.getSystemInfoSync();
      screenWidth.value = sysInfo.windowWidth || 375;
      screenHeight.value = sysInfo.windowHeight || 667;
    } catch (error) {
      console.warn("[useLayout] Failed to get system info:", error);
    }
  };

  // ==================== 侧边栏控制 ====================

  /** 打开侧边栏 */
  const openSidebar = () => {
    sidebarVisible.value = true;
  };

  /** 关闭侧边栏 */
  const closeSidebar = () => {
    sidebarVisible.value = false;
  };

  /** 切换侧边栏 */
  const toggleSidebar = () => {
    sidebarVisible.value = !sidebarVisible.value;
  };

  // ==================== 生命周期 ====================

  onMounted(() => {
    updateScreenSize();

    // #ifdef H5
    // H5 环境监听窗口变化
    window.addEventListener("resize", updateScreenSize);
    // #endif
  });

  onUnmounted(() => {
    // #ifdef H5
    window.removeEventListener("resize", updateScreenSize);
    // #endif
  });

  return {
    // 屏幕尺寸
    screenWidth,
    screenHeight,

    // 响应式断点
    isWideScreen,
    isTablet,
    isDesktop,
    isLargeDesktop,
    isMobile,
    isLandscape,

    // 侧边栏控制
    sidebarVisible,
    openSidebar,
    closeSidebar,
    toggleSidebar,

    // 工具方法
    updateScreenSize,
  };
}

/**
 * 获取当前页面路由
 */
export function useCurrentRoute() {
  const currentRoute = ref("/pages/home/index");

  const updateCurrentRoute = () => {
    const pages = getCurrentPages();
    if (pages.length > 0) {
      const currentPage = pages[pages.length - 1];
      currentRoute.value = "/" + (currentPage?.route || "pages/home/index");
    }
  };

  onMounted(() => {
    updateCurrentRoute();
  });

  return {
    currentRoute,
    updateCurrentRoute,
  };
}
