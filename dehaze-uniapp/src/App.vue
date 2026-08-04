<script lang="ts" setup>
import { onHide, onLaunch, onShow } from "@dcloudio/uni-app";
import { useAuthStore } from "@/store/auth";
import { setupRouteGuard, checkInitialAuth } from "@/routers/guard";

onLaunch(() => {
  // 1. 安装路由守卫：拦截未登录跳转（不拦截首次启动自动加载的首页）
  setupRouteGuard();

  // 2. 初始化认证状态：从 storage 恢复登录态，并注册会话失效监听
  const authStore = useAuthStore();
  authStore.init();

  // 3. 显式检查初始登录态：补充守卫不覆盖的首页首启场景
  checkInitialAuth();
});

onShow(() => {});

onHide(() => {});
</script>

<style lang="scss">
@import "uview-plus/index.scss";

/* 全局样式 */
/* variables.scss 已通过 vite.config.ts 的 scss.additionalData 全局注入，无需在此重复导入 */
@import "@/styles/common.scss";
</style>
