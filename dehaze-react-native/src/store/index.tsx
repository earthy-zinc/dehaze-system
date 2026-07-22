/**
 * 全局状态管理入口
 *
 * 组合所有 Context Provider，统一在 App.tsx 包裹。
 * 页面间数据通过路由参数传递，全局仅保留认证状态。
 */
import { AuthProvider } from './AuthContext';
import type { ReactNode } from 'react';

export { AuthProvider, useAuth } from './AuthContext';

export function AppProviders({ children }: { children: ReactNode }) {
  return <AuthProvider>{children}</AuthProvider>;
}
