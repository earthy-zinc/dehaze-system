export const typography = {
  sizes: {
    hero: 48,
    h1: 40,
    h2: 36,
    h3: 28,
    h4: 24,
    h5: 20,
    h6: 18,
    bodyLarge: 18,
    body: 16,
    bodySmall: 14,
    caption: 13,
    tiny: 12,
    small: 13,
    medium: 16,
    large: 18,
  },
  /** 语义化别名（供组件使用） */
  small: 13,
  medium: 16,
  large: 18,
  weights: {
    regular: '400' as '400',
    medium: '500' as '500',
    semibold: '600' as '600',
    bold: '700' as '700',
  },
  lineHeights: {
    hero: 1.1,
    title: 1.2,
    body: 1.6,
    compact: 1.4,
  },
  letterSpacing: {
    tight: -1,
    normal: -0.5,
    wide: 0.5,
  },
} as const;

export type Typography = typeof typography;
