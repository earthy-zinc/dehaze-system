export const gradientColors = {
  primary: ['#3b82f6', '#6366f1'],
  metric: ['#667eea', '#764ba2'],
};

export const colors = {
  primary: '#3b82f6',
  primaryLight: '#eff6ff',
  primaryDark: '#1d4ed8',
  secondary: '#14b8a6',
  secondaryLight: '#f0fdfa',
  gradient: gradientColors,
  status: {
    success: '#4caf50',
    warning: '#ff9800',
    error: '#f44336',
    info: '#2196f3',
  },
  badge: {
    success: { bg: 'rgba(76,175,80,0.12)', text: '#2e7d32' },
    warning: { bg: 'rgba(255,152,0,0.12)', text: '#ef6c00' },
    error: { bg: 'rgba(244,67,54,0.12)', text: '#c62828' },
    info: { bg: 'rgba(33,150,243,0.12)', text: '#1565c0' },
  },
  text: {
    primary: '#1f2937',
    secondary: '#6b7280',
    tertiary: '#9ca3af',
    muted: '#9ca3af',
    inverse: '#ffffff',
    link: '#3b82f6',
  },
  background: {
    primary: '#ffffff',
    secondary: '#f9fafb',
    tertiary: '#f3f4f6',
    overlay: 'rgba(0, 0, 0, 0.75)',
    translucent: 'rgba(255, 255, 255, 0.15)',
  },
  border: {
    light: '#e5e7eb',
    transparent: 'transparent',
  },
} as const;

export type Colors = typeof colors;
