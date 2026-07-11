<script lang="ts" setup>
import { onHide, onLaunch, onShow } from "@dcloudio/uni-app";
import { useAuthStore } from "@/store/auth";
import { useUserStore } from "@/store/user";
import { setupRouteGuard } from "@/routers/guard";

onLaunch(() => {
  console.log("[App] Launch");

  // 安装路由守卫
  setupRouteGuard();

  // 初始化认证状态
  const authStore = useAuthStore();
  authStore.init();

  // 初始化用户偏好
  const userStore = useUserStore();
  userStore.initPreferences();
});

onShow(() => {
  console.log("[App] Show");
});

onHide(() => {
  console.log("[App] Hide");
});
</script>
<style lang="scss">
@import "uview-plus/index.scss";

/* 全局样式 */
@import "@/styles/variables.scss";
@import "@/styles/common.scss";
</style>
