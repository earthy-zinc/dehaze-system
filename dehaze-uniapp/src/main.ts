import { createPinia } from "pinia";
import { createSSRApp } from "vue";
import App from "./App.vue";
import { Logger } from "dehaze-sdk-js";
// 初始化 SDK（注入 uni.request adapter、token、baseURL 等）
import "./api/sdk-setup";
// SVG 图标 sprite 注册（配合 vite-plugin-svg-icons，由 src/assets/icons/*.svg 自动生成）
import "virtual:svg-icons-register";

export function createApp() {
  const app = createSSRApp(App);

  // 前端日志监控：Vue 渲染/生命周期异常转发给 Logger 上报（error_type=js）
  app.config.errorHandler = (err, _instance, info) => {
    Logger.getInstance()?.error(
      `Vue 应用异常: ${(err as Error)?.message ?? String(err)}`,
      {
        error_type: "js",
        error_source: "vue_error_handler",
        error_stack: `${(err as Error)?.stack ?? ""}\ninfo: ${info ?? ""}`,
      }
    );
  };

  // Pinia 状态管理
  const pinia = createPinia();
  app.use(pinia);

  return {
    app,
    pinia,
  };
}
