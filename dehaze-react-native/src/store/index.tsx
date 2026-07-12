/**
 * 全局状态管理入口
 *
 * 组合所有 Context Provider，统一在 App.tsx 包裹。
 */
import { AlgorithmProvider } from './AlgorithmContext';
import { AuthProvider } from './AuthContext';
import { ImageProvider } from './ImageContext';
import type { ReactNode } from 'react';

export { AuthProvider, useAuth } from './AuthContext';
export { AlgorithmProvider, useAlgorithm } from './AlgorithmContext';
export { ImageProvider, useImageContext } from './ImageContext';

export function AppProviders({ children }: { children: ReactNode }) {
  return (
    <AuthProvider>
      <AlgorithmProvider>
        <ImageProvider>{children}</ImageProvider>
      </AlgorithmProvider>
    </AuthProvider>
  );
}
