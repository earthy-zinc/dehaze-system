/**
 * 布局相关 Composable
 * 提供侧边栏控制
 */
import { ref, onMounted } from "vue";

/**
 * 布局 Hook
 * 提供侧边栏控制
 */
export function useLayout() {
  /** 侧边栏可见状态 */
  const sidebarVisible = ref(false);

  /** 关闭侧边栏 */
  const closeSidebar = () => {
    sidebarVisible.value = false;
  };

  /** 切换侧边栏 */
  const toggleSidebar = () => {
    sidebarVisible.value = !sidebarVisible.value;
  };

  return {
    sidebarVisible,
    closeSidebar,
    toggleSidebar,
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
