<script lang="ts" setup>
import { onHide, onLaunch, onShow } from "@dcloudio/uni-app";
import { useAuthStore } from "@/store/auth";
import { setupRouteGuard, checkInitialAuth } from "@/routers/guard";

onLaunch(() => {
  // 安装路由守卫
  setupRouteGuard();

  // 初始化认证状态
  const authStore = useAuthStore();
  authStore.init();

  // 检查初始登录态：若未登录且当前页需要登录，则跳转登录页
  // uni-app 的 addInterceptor 不会拦截应用首次启动时自动加载的首页，因此需要显式检查
  // 放在 init() 之后，确保 token 已从 storage 恢复
  checkInitialAuth();
});

onShow(() => {});

onHide(() => {});
</script>
<style lang="scss">
@import "uview-plus/index.scss";

/* 全局样式 */
@import "@/styles/variables.scss";
@import "@/styles/common.scss";
</style>
