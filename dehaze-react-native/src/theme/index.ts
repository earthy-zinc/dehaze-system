import { colors } from './colors';
import { typography } from './typography';
import { spacing, layout } from './spacing';

export const theme = {
  colors,
  typography,
  spacing,
  layout,
} as const;

export type Theme = typeof theme;

export * from './colors';
export * from './typography';
export * from './spacing';
