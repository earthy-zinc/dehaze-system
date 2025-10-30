import * as ElementPlusIconsVue from "@element-plus/icons-vue";
import type { App } from "vue";

// 注册所有图标
export function setupElIcons(app: App<Element>) {
  for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
    app.component(key, component);
  }
}
