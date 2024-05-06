<template>
  <div :class="{ 'has-logo': sidebarLogo }">
    <!--混合布局-->
    <div class="flex w-full" v-if="layout == LayoutEnum.MIX">
      <SidebarLogo v-if="sidebarLogo" :collapse="!appStore.sidebar.opened" />
      <SidebarMixTopMenu class="flex-1" />
      <NavbarRight />
    </div>
    <!--左侧布局 || 顶部布局 -->
    <template v-else>
      <SidebarLogo v-if="sidebarLogo" :collapse="!appStore.sidebar.opened" />
      <el-scrollbar>
        <SidebarMenu :menu-list="permissionStore.routes" base-path="" />
      </el-scrollbar>
      <NavbarRight v-if="layout === LayoutEnum.TOP" />
    </template>
  </div>
</template>

<script setup lang="ts">
import { useSettingsStore, usePermissionStore, useAppStore } from "@/store";
import { LayoutEnum } from "@/enums/LayoutEnum";

const appStore = useAppStore();
const settingsStore = useSettingsStore();
const permissionStore = usePermissionStore();

const sidebarLogo = computed(() => settingsStore.sidebarLogo);
const layout = computed(() => settingsStore.layout);
</script>

<style lang="scss" scoped>
.has-logo {
  .el-scrollbar {
    height: calc(100vh - $navbar-height);
  }
}
</style>

<style lang="scss">
:root {
  --el-menu-base-level-padding: 18px;
  --el-menu-icon-width: 18px;
}
$el-menu-base-level-padding: var(--el-menu-base-level-padding);
$el-menu-icon-width: var(--el-menu-icon-width);

.el-sub-menu {
  &__title {
    &:hover {
      background: $menu-hover;
    }

    padding: 0 0 0 $el-menu-base-level-padding;
  }

  &__icon-arrow {
    margin-right: -6px;
  }
}

.el-menu {
  &--horizontal {
    justify-content: center;

    .el-menu-item:not(.is-disabled) {
      &:focus,
      &:hover {
        background: $menu-hover;
      }
    }
  }

  &:not(.el-menu--collapse) {
    .el-sub-menu__title {
      padding-right: calc(
        #{$el-menu-base-level-padding} + #{$el-menu-icon-width} - 8px
      );
    }
  }
}
</style>
