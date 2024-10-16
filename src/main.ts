import { createApp } from "vue";
import App from "./App.vue";
import router from "@/router";
import { setupStore } from "@/store";
import { setupDirective } from "@/directive";
import { setupElIcons, setupI18n, setupPermission } from "@/plugins";
import VueViewer from "v-viewer";
import "viewerjs/dist/viewer.css";
import VueLazyLoad from "vue3-lazyload";

// 本地SVG图标
import "virtual:svg-icons-register";

// 样式
import "element-plus/theme-chalk/dark/css-vars.css";
import "@/styles/index.scss";
import "uno.css";
import "animate.css";

const app = createApp(App);
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
