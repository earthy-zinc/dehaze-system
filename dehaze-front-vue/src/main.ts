import { setupDirective } from "@/directive";
import { setupElIcons, setupI18n, setupPermission } from "@/plugins";
import router from "@/router";
import { setupStore } from "@/store";
import setupRequest from "@/utils/request";
import { Logger, ConsoleTransport, RemoteTransport } from "dehaze-sdk-js";
import VueViewer from "v-viewer";
import "viewerjs/dist/viewer.css";
import { createApp } from "vue";
import VueLazyLoad from "vue3-lazyload";

// 本地SVG图标
import "virtual:svg-icons-register";

// 样式
import "element-plus/theme-chalk/dark/css-vars.css";
import "@/styles/index.scss";
import "uno.css";
import App from "./App.vue";
import "animate.css";

const app = createApp(App);
// 前端日志监控：注册全局错误捕获 + 离线上报。SDK 不感知环境，
// 由应用端按构建产物组装 transports（开发仅 Console，生产追加 Remote）
Logger.install({
  app: "vue",
  appVersion: __APP_INFO__.pkg.version,
  transports: import.meta.env.PROD
    ? [new ConsoleTransport(), new RemoteTransport()]
    : [new ConsoleTransport()],
});
setupRequest();
// Vue 渲染/生命周期异常转发给 Logger 上报（error_type=js）
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
// 全局注册 自定义指令(directive)
setupDirective(app);
// 全局注册 状态管理(store)
setupStore(app);
// 全局注册Element-plus图标
setupElIcons(app);
// 国际化
setupI18n(app);
// 注册动态路由
setupPermission();
app
  .use(router)
  .use(VueLazyLoad, {
    // options...
  })
  .use(VueViewer, {
    defaultOptions: {
      inline: true,
      button: true, //右上角按钮
      navbar: true, //底部缩略图
      title: true, //当前图片标题
      toolbar: true, //底部工具栏
      tooltip: true, //显示缩放百分比
      movable: true, //是否可以移动
      zoomable: true, //是否可以缩放
      rotatable: true, //是否可旋转
      scalable: true, //是否可翻转
      transition: true, //使用 CSS3 过度
      fullscreen: true, //播放时是否全屏
      keyboard: true, //是否支持键盘
    },
  })
  .mount("#app");
