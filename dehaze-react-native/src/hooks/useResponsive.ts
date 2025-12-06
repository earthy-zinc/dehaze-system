import { useState, useEffect, useCallback } from 'react';
import { Dimensions } from 'react-native';

// 断点定义（参考 CSS 媒体查询标准）
export const BREAKPOINTS = {
  xs: 0, // 小手机
  sm: 375, // 标准手机
  md: 768, // 平板
  lg: 1024, // 大平板/小桌面
  xl: 1280, // 桌面
} as const;

export type BreakpointKey = keyof typeof BREAKPOINTS;

interface ResponsiveInfo {
  width: number;
  height: number;
  isPortrait: boolean;
  isLandscape: boolean;
  // 断点状态
  breakpoint: BreakpointKey;
  isXs: boolean;
  isSm: boolean;
  isMd: boolean;
  isLg: boolean;
  isXl: boolean;
  // 设备类型
  isMobile: boolean;
  isTablet: boolean;
  isDesktop: boolean;
  // 响应式列数
  columns: number;
  // 响应式间距
  spacing: number;
  // 响应式字体缩放
  fontScale: number;
  // 响应式内边距
  containerPadding: number;
}

/**
 * 响应式布局 Hook
 * 提供跨端响应式设计支持
 */
export function useResponsive(): ResponsiveInfo {
  const [dimensions, setDimensions] = useState(() => Dimensions.get('window'));

  useEffect(() => {
    const subscription = Dimensions.addEventListener('change', ({ window }) => {
      setDimensions(window);
    });

    return () => subscription?.remove();
  }, []);

  const { width, height } = dimensions;

  // 计算当前断点
  const getBreakpoint = useCallback((): BreakpointKey => {
    if (width >= BREAKPOINTS.xl) return 'xl';
    if (width >= BREAKPOINTS.lg) return 'lg';
    if (width >= BREAKPOINTS.md) return 'md';
    if (width >= BREAKPOINTS.sm) return 'sm';
    return 'xs';
  }, [width]);

  const breakpoint = getBreakpoint();

  // 计算响应式列数（用于网格布局）
  const getColumns = useCallback((): number => {
    if (width >= BREAKPOINTS.xl) return 4;
    if (width >= BREAKPOINTS.lg) return 3;
    if (width >= BREAKPOINTS.md) return 3;
    if (width >= BREAKPOINTS.sm) return 2;
    return 2;
  }, [width]);

  // 计算响应式间距
  const getSpacing = useCallback((): number => {
    if (width >= BREAKPOINTS.lg) return 32;
    if (width >= BREAKPOINTS.md) return 24;
    return 16;
  }, [width]);

  // 计算字体缩放因子
  const getFontScale = useCallback((): number => {
    if (width >= BREAKPOINTS.xl) return 1.1;
    if (width >= BREAKPOINTS.lg) return 1.05;
    if (width >= BREAKPOINTS.md) return 1;
    if (width >= BREAKPOINTS.sm) return 1;
    return 0.9;
  }, [width]);

  // 计算容器内边距
  const getContainerPadding = useCallback((): number => {
    if (width >= BREAKPOINTS.xl) return 40;
    if (width >= BREAKPOINTS.lg) return 32;
    if (width >= BREAKPOINTS.md) return 24;
    return 20;
  }, [width]);

  return {
    width,
    height,
    isPortrait: height > width,
    isLandscape: width > height,
    breakpoint,
    isXs: breakpoint === 'xs',
    isSm: breakpoint === 'sm',
    isMd: breakpoint === 'md',
    isLg: breakpoint === 'lg',
    isXl: breakpoint === 'xl',
    isMobile: width < BREAKPOINTS.md,
    isTablet: width >= BREAKPOINTS.md && width < BREAKPOINTS.lg,
    isDesktop: width >= BREAKPOINTS.lg,
    columns: getColumns(),
    spacing: getSpacing(),
    fontScale: getFontScale(),
    containerPadding: getContainerPadding(),
  };
}

/**
 * 响应式值选择器
 * 根据当前断点返回对应的值
 */
export function useResponsiveValue<T>(
  values: Partial<Record<BreakpointKey, T>>,
  defaultValue: T,
): T {
  const { breakpoint } = useResponsive();

  // 按优先级查找值：当前断点 -> 更小的断点 -> 默认值
  const breakpointOrder: BreakpointKey[] = ['xl', 'lg', 'md', 'sm', 'xs'];
  const currentIndex = breakpointOrder.indexOf(breakpoint);

  for (let i = currentIndex; i < breakpointOrder.length; i++) {
    const key = breakpointOrder[i];
    if (values[key] !== undefined) {
      return values[key] as T;
    }
  }

  return defaultValue;
}

/**
 * 响应式样式生成器
 * 根据断点生成不同的样式
 */
export function createResponsiveStyles<T extends Record<string, any>>(
  baseStyles: T,
  responsiveStyles: Partial<Record<BreakpointKey, Partial<T>>>,
): (breakpoint: BreakpointKey) => T {
  return (breakpoint: BreakpointKey) => {
    const breakpointOrder: BreakpointKey[] = ['xs', 'sm', 'md', 'lg', 'xl'];
    const currentIndex = breakpointOrder.indexOf(breakpoint);

    let mergedStyles = { ...baseStyles };

    for (let i = 0; i <= currentIndex; i++) {
      const key = breakpointOrder[i];
      if (responsiveStyles[key]) {
        mergedStyles = { ...mergedStyles, ...responsiveStyles[key] };
      }
    }

    return mergedStyles;
  };
}

export default useResponsive;
