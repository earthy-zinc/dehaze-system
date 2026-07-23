/**
 * 对比页控制栏共享样式
 *
 * Overlay / Magnifier / Filter 三个对比页的控制栏样式完全一致，
 * Filter 额外使用 active / disabled 变体。
 */
import { StyleSheet } from 'react-native';
import { theme } from '@/theme';

export const controlBarStyles = StyleSheet.create({
  bar: {
    flexDirection: 'row',
    gap: theme.spacing.sm,
    paddingHorizontal: theme.spacing.md,
    paddingVertical: theme.spacing.sm,
    backgroundColor: theme.colors.background.primary,
    borderBottomWidth: 1,
    borderBottomColor: theme.colors.border.light,
  },
});

export const controlButtonStyles = StyleSheet.create({
  button: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: theme.spacing.md,
    paddingVertical: theme.spacing.xs,
    borderRadius: theme.layout.borderRadius.full,
    backgroundColor: theme.colors.background.tertiary,
  },
  buttonActive: {
    backgroundColor: theme.colors.primary,
  },
  buttonDisabled: {
    opacity: 0.4,
  },
  text: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.secondary,
  },
  textActive: {
    color: '#fff',
  },
});
