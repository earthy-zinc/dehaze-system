<!-- 左侧边菜单：包括左侧布局(left)、顶部布局(all)、混合布局(left) -->
<template>
  <el-menu
    :active-text-color="variables['menu-active-text']"
    :background-color="variables['menu-background']"
    :collapse="!appStore.sidebar.opened"
    :collapse-transition="false"
    :default-active="currentRoute.path"
    :mode="layout === 'top' ? 'horizontal' : 'vertical'"
    :text-color="variables['menu-text']"
    :unique-opened="false"
  >
    <SidebarMenuItem
      v-for="route in menuList"
      :key="route.path"
      :base-path="resolvePath(route.path)"
      :is-collapse="!appStore.sidebar.opened"
      :item="route"
    />
  </el-menu>
</template>

<script lang="ts" setup>
import { useAppStore, useSettingsStore } from "@/store";
import variables from "@/styles/variables.module.scss";
import { isExternal } from "@/utils/index";
import path from "path-browserify";

const settingsStore = useSettingsStore();
const appStore = useAppStore();
const currentRoute = useRoute();
const layout = computed(() => settingsStore.layout);
const props = defineProps({
  menuList: {
    required: true,
    default: () => {
      return [];
    },
    type: Array<any>,
  },
  basePath: {
    type: String,
    required: true,
  },
});

/**
 * 解析路径
 *
 * @param routePath 路由路径 /user
 */
function resolvePath(routePath: string) {
  if (isExternal(routePath)) {
    return routePath;
  }
  if (isExternal(props.basePath)) {
    return props.basePath;
  }

  // 完整绝对路径 = 父级路径(/system) + 路由路径(/user)
  const fullPath = path.resolve(props.basePath, routePath);
  return fullPath;
}
</script>
