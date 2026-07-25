<template>
  <div class="titlebar">
    <div class="titlebar__left">
      <img :src="logo" class="titlebar__logo" />
      <span class="titlebar__title">{{ defaultSettings.title }}</span>
    </div>
    <div class="titlebar__controls">
      <button class="titlebar__btn" title="最小化" @click="handleMinimize">
        <svg width="10" height="10" viewBox="0 0 10 10">
          <path d="M0 5h10" stroke="currentColor" stroke-width="1" />
        </svg>
      </button>
      <button
        class="titlebar__btn"
        title="最大化"
        @click="handleToggleMaximize"
      >
        <svg v-if="!isMaximized" width="10" height="10" viewBox="0 0 10 10">
          <rect
            x="0.5"
            y="0.5"
            width="9"
            height="9"
            fill="none"
            stroke="currentColor"
            stroke-width="1"
          />
        </svg>
        <svg v-else width="10" height="10" viewBox="0 0 10 10">
          <rect
            x="0.5"
            y="2.5"
            width="7"
            height="7"
            fill="none"
            stroke="currentColor"
            stroke-width="1"
          />
          <rect
            x="2.5"
            y="0.5"
            width="7"
            height="7"
            fill="none"
            stroke="currentColor"
            stroke-width="1"
          />
        </svg>
      </button>
      <button
        class="titlebar__btn titlebar__btn--close"
        title="关闭"
        @click="handleClose"
      >
        <svg width="10" height="10" viewBox="0 0 10 10">
          <path
            d="M0 0l10 10M10 0L0 10"
            stroke="currentColor"
            stroke-width="1"
          />
        </svg>
      </button>
    </div>
  </div>
</template>

<script lang="ts" setup>
import defaultSettings from "@/settings";

const logo = ref("/favicon.ico");
const isMaximized = ref(false);

function handleMinimize() {
  window.electronAPI?.minimize();
}

function handleToggleMaximize() {
  window.electronAPI?.toggleMaximize();
  isMaximized.value = !isMaximized.value;
}

function handleClose() {
  window.electronAPI?.close();
}
</script>

<style lang="scss" scoped>
.titlebar {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 2000;
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: $titlebar-height;
  padding-left: 12px;
  background: var(--el-bg-color);
  border-bottom: 1px solid var(--el-border-color-light);
  -webkit-app-region: drag;
  user-select: none;
}

.titlebar__left {
  display: flex;
  align-items: center;
  gap: 8px;
}

.titlebar__logo {
  width: 18px;
  height: 18px;
}

.titlebar__title {
  font-size: 12px;
  font-weight: 500;
  color: var(--el-text-color-primary);
}

.titlebar__controls {
  display: flex;
  height: 100%;
  -webkit-app-region: no-drag;
}

.titlebar__btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 46px;
  height: 100%;
  padding: 0;
  border: none;
  background: transparent;
  color: var(--el-text-color-regular);
  cursor: pointer;
  transition: background 0.15s;

  &:hover {
    background: var(--el-fill-color-light);
  }

  &:active {
    background: var(--el-fill-color);
  }
}

.titlebar__btn--close:hover {
  background: #e81123;
  color: #fff;
}
</style>
