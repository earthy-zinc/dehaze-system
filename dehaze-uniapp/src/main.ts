import uviewPlus from "uview-plus";
import { createPinia } from "pinia";
import { createSSRApp } from "vue";
import App from "./App.vue";

export function createApp() {
  const app = createSSRApp(App);

  // Pinia 状态管理
  const pinia = createPinia();
  app.use(pinia);

  // uview-plus UI 组件库
  app.use(uviewPlus);

  return {
    app,
    pinia,
  };
}
