export const colors = {
  primary: '#3b82f6',
  primaryLight: '#eff6ff',
  primaryDark: '#1e40af',
  secondary: '#14b8a6',
  secondaryLight: '#f0fdfa',
  status: {
    success: '#34d399',
    warning: '#fbbf24',
    error: '#ef4444',
    info: '#3b82f6',
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
