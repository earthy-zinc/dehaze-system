/**
 * 用户状态管理
 *
 * 扩展用途：
 * - 跨页面共享的用户偏好设置
 * - 用户相关操作的状态缓存
 * - 当前处理中的图片/算法等临时状态
 */

import { defineStore } from "pinia";
import { ref } from "vue";

export const useUserStore = defineStore("user", () => {
  // ==================== 状态 ====================

  /** 展示模式偏好（数据集图片 grid / waterfall） */
  const displayMode = ref<"grid" | "waterfall">("grid");

  /** 主题模式（预留） */
  const themeMode = ref<"light" | "dark">("light");

  // ==================== 方法 ====================

  /** 初始化用户偏好 */
  function initPreferences() {
    try {
      const savedMode = uni.getStorageSync("display_mode");
      if (savedMode === "grid" || savedMode === "waterfall") {
        displayMode.value = savedMode;
      }
    } catch {
      // 忽略读取错误
    }
  }

  /** 设置展示模式 */
  function setDisplayMode(mode: "grid" | "waterfall") {
    displayMode.value = mode;
    try {
      uni.setStorageSync("display_mode", mode);
    } catch {
      // 忽略写入错误
    }
  }

  return {
    displayMode,
    themeMode,
    initPreferences,
    setDisplayMode,
  };
});
