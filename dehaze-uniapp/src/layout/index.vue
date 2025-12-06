<template>
  <view class="page-layout">
    <!-- 顶部导航栏 -->
    <AppNavbar @toggle-menu="toggleSidebar" @search="handleSearch" />

    <!-- 侧边菜单 -->
    <AppSidebar
      :visible="sidebarVisible"
      :current-route="currentRoute"
      @close="closeSidebar"
      @navigate="handleNavigate"
    />

    <!-- 主内容区 -->
    <view class="page-content" :class="{ 'with-tabbar': showTabbar }">
      <slot />
    </view>

    <!-- 底部导航栏 -->
    <AppTabbar v-if="showTabbar" :current-route="currentRoute" />
  </view>
</template>

<script lang="ts" setup>
import { ref, onMounted } from "vue";
import AppNavbar from "./Navbar.vue";
import AppSidebar from "./Sidebar.vue";
import AppTabbar from "./Tabbar.vue";
import { useLayout, useCurrentRoute } from "@/composables/useLayout";

interface Props {
  /** 是否显示底部导航栏 */
  showTabbar?: boolean;
  /** 页面标题（用于搜索等场景） */
  title?: string;
}

interface Emits {
  (e: "search"): void;
  (e: "navigate", route: string): void;
}

const props = withDefaults(defineProps<Props>(), {
  showTabbar: true,
  title: "",
});

const emit = defineEmits<Emits>();

// 布局状态
const { sidebarVisible, toggleSidebar, closeSidebar } = useLayout();
const { currentRoute, updateCurrentRoute } = useCurrentRoute();

onMounted(() => {
  updateCurrentRoute();
});

// 搜索处理
const handleSearch = () => {
  emit("search");
};

// 导航处理
const handleNavigate = (route: string) => {
  emit("navigate", route);
};
</script>

<style lang="scss" scoped>
.page-layout {
  width: 100%;
  min-height: 100vh;
  background: #ffffff;
}

.page-content {
  width: 100%;

  &.with-tabbar {
    // 为底部导航栏留出空间
    padding-bottom: calc(100rpx + constant(safe-area-inset-bottom));
    padding-bottom: calc(100rpx + env(safe-area-inset-bottom));
  }
}
</style>
