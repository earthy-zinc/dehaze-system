/**
 * 全局状态管理入口
 *
 * zustand stores，无需 Provider 包裹。
 * 页面间数据通过路由参数传递，全局仅保留认证状态。
 */
export { useAuthStore } from './auth';
