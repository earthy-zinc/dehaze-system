import { StyleSheet } from 'react-native';
import { theme } from '@/theme';

export const styles = StyleSheet.create({
  screenContainer: {
    flex: 1,
    backgroundColor: theme.colors.background.secondary,
  },
  container: {
    flex: 1,
    backgroundColor: theme.colors.background.secondary,
  },
  stateContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: theme.colors.background.secondary,
  },
  // 章节锚点导航
  sectionNav: {
    backgroundColor: theme.colors.background.primary,
    borderBottomWidth: 1,
    borderBottomColor: theme.colors.border.light,
    maxHeight: 52,
  },
  sectionNavContent: {
    paddingHorizontal: theme.spacing.md,
    paddingVertical: theme.spacing.sm,
    gap: theme.spacing.sm,
  },
  sectionChip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: theme.layout.borderRadius.full,
    backgroundColor: theme.colors.background.tertiary,
    marginRight: 4,
  },
  sectionChipActive: {
    backgroundColor: theme.colors.primary,
  },
  sectionChipText: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.secondary,
    fontWeight: theme.typography.weights.medium,
  },
  sectionChipTextActive: {
    color: '#fff',
    fontWeight: theme.typography.weights.semibold,
  },
  // 滚动容器
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    paddingBottom: theme.spacing.xl,
  },
  bottomSpacer: {
    height: 100,
  },
  // Hero
  hero: {
    padding: theme.spacing.lg,
    paddingTop: theme.spacing.xl,
    paddingBottom: theme.spacing.xl,
    marginHorizontal: theme.spacing.md,
    marginTop: theme.spacing.md,
    borderRadius: theme.layout.borderRadius.xxl,
    ...theme.layout.shadows.lg,
  },
  heroTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: theme.spacing.md,
  },
  heroIconWrap: {
    width: 56,
    height: 56,
    borderRadius: theme.layout.borderRadius.lg,
    backgroundColor: 'rgba(255,255,255,0.2)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  heroBadges: {
    flexDirection: 'row',
    gap: 6,
    flexWrap: 'wrap',
    justifyContent: 'flex-end',
  },
  heroBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: theme.layout.borderRadius.full,
    backgroundColor: 'rgba(255,255,255,0.25)',
  },
  heroBadgeText: {
    fontSize: theme.typography.sizes.small,
    fontWeight: theme.typography.weights.semibold,
  },
  heroBadgeTextLight: {
    fontSize: theme.typography.sizes.small,
    color: '#fff',
    fontWeight: theme.typography.weights.semibold,
  },
  heroTitle: {
    fontSize: 26,
    fontWeight: theme.typography.weights.bold,
    color: '#fff',
    marginBottom: 6,
    letterSpacing: -0.5,
  },
  heroSubtitle: {
    fontSize: theme.typography.sizes.medium,
    color: 'rgba(255,255,255,0.85)',
    marginBottom: theme.spacing.sm,
    fontWeight: theme.typography.weights.medium,
  },
  heroDesc: {
    fontSize: theme.typography.sizes.bodySmall,
    color: 'rgba(255,255,255,0.75)',
    lineHeight: 20,
    marginBottom: theme.spacing.md,
  },
  heroMetrics: {
    flexDirection: 'row',
    backgroundColor: 'rgba(255,255,255,0.15)',
    borderRadius: theme.layout.borderRadius.lg,
    paddingVertical: theme.spacing.sm,
  },
  heroMetricItem: {
    flex: 1,
    alignItems: 'center',
  },
  heroMetricValue: {
    fontSize: 20,
    fontWeight: theme.typography.weights.bold,
    color: '#fff',
  },
  heroMetricLabel: {
    fontSize: 11,
    color: 'rgba(255,255,255,0.8)',
    marginTop: 2,
  },
  heroMetricDivider: {
    width: 1,
    backgroundColor: 'rgba(255,255,255,0.2)',
    marginVertical: 4,
  },
  // 章节
  sectionWrap: {
    marginTop: theme.spacing.lg,
    paddingHorizontal: theme.spacing.md,
  },
  sectionTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginBottom: theme.spacing.sm,
    paddingHorizontal: 4,
  },
  sectionTitleIcon: {
    width: 26,
    height: 26,
    borderRadius: theme.layout.borderRadius.sm,
    backgroundColor: `${theme.colors.primary}15`,
    justifyContent: 'center',
    alignItems: 'center',
  },
  sectionTitleText: {
    fontSize: theme.typography.sizes.medium,
    fontWeight: theme.typography.weights.bold,
    color: theme.colors.text.primary,
    letterSpacing: 0.3,
  },
  // 卡片
  card: {
    backgroundColor: theme.colors.background.primary,
    borderRadius: theme.layout.borderRadius.lg,
    padding: theme.spacing.md,
    ...theme.layout.shadows.sm,
  },
  // 信息行
  infoRow: {
    flexDirection: 'row',
    paddingVertical: 10,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: theme.colors.border.light,
  },
  infoLabel: {
    width: 80,
    fontSize: theme.typography.sizes.bodySmall,
    color: theme.colors.text.tertiary,
    fontWeight: theme.typography.weights.medium,
  },
  infoValue: {
    flex: 1,
    fontSize: theme.typography.sizes.bodySmall,
    color: theme.colors.text.primary,
    fontWeight: theme.typography.weights.medium,
  },
  infoValueMono: {
    fontFamily: 'Menlo',
    fontSize: 12,
    color: theme.colors.text.secondary,
  },
  // 描述
  descriptionText: {
    fontSize: theme.typography.sizes.bodySmall,
    color: theme.colors.text.secondary,
    lineHeight: 22,
  },
  emptyInlineText: {
    fontSize: theme.typography.sizes.bodySmall,
    color: theme.colors.text.tertiary,
    textAlign: 'center',
    paddingVertical: theme.spacing.md,
  },
  // 效果样例
  sampleHint: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.tertiary,
    marginBottom: theme.spacing.md,
  },
  sampleCompareRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  sampleImageBox: {
    flex: 1,
  },
  sampleImagePlaceholder: {
    aspectRatio: 1,
    borderRadius: theme.layout.borderRadius.md,
    backgroundColor: theme.colors.background.tertiary,
    justifyContent: 'center',
    alignItems: 'center',
    gap: 6,
    borderWidth: 1,
    borderColor: theme.colors.border.light,
  },
  sampleImagePlaceholderClean: {
    backgroundColor: '#ECFDF5',
    borderColor: '#A7F3D0',
  },
  sampleImageLabel: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.secondary,
    fontWeight: theme.typography.weights.medium,
  },
  sampleArrow: {
    paddingHorizontal: 2,
  },
  sampleNote: {
    fontSize: 11,
    color: theme.colors.text.tertiary,
    marginTop: theme.spacing.sm,
    textAlign: 'center',
  },
  // 参数代码块
  codeBlock: {
    backgroundColor: '#0F172A',
    borderRadius: theme.layout.borderRadius.md,
    padding: theme.spacing.md,
  },
  codeText: {
    fontFamily: 'Menlo',
    fontSize: 12,
    color: '#E2E8F0',
    lineHeight: 18,
  },
  // 指标条
  metricBarWrap: {
    marginBottom: theme.spacing.md,
  },
  metricBarHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 6,
  },
  metricLabel: {
    fontSize: theme.typography.sizes.bodySmall,
    color: theme.colors.text.secondary,
    fontWeight: theme.typography.weights.medium,
  },
  metricValue: {
    fontSize: theme.typography.sizes.medium,
    fontWeight: theme.typography.weights.bold,
  },
  metricSuffix: {
    fontSize: theme.typography.sizes.small,
    fontWeight: theme.typography.weights.regular,
    color: theme.colors.text.tertiary,
    marginLeft: 2,
  },
  metricBarTrack: {
    height: 8,
    backgroundColor: theme.colors.background.tertiary,
    borderRadius: 4,
    overflow: 'hidden',
  },
  metricBarFill: {
    height: '100%',
    borderRadius: 4,
  },
  // 版本时间线
  timeline: {
    paddingTop: 4,
  },
  timelineItem: {
    flexDirection: 'row',
  },
  timelineLeft: {
    width: 20,
    alignItems: 'center',
  },
  timelineDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: theme.colors.text.tertiary,
    marginTop: 4,
  },
  timelineDotActive: {
    backgroundColor: theme.colors.primary,
    width: 12,
    height: 12,
    borderRadius: 6,
    marginTop: 3,
    borderWidth: 2,
    borderColor: '#fff',
    shadowColor: theme.colors.primary,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.5,
    shadowRadius: 4,
    elevation: 4,
  },
  timelineLine: {
    flex: 1,
    width: 1,
    backgroundColor: theme.colors.border.light,
    marginTop: 4,
    marginBottom: 0,
    minHeight: 40,
  },
  timelineContent: {
    flex: 1,
    paddingBottom: theme.spacing.md,
    marginLeft: 8,
  },
  timelineHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginBottom: 4,
  },
  timelineVersion: {
    fontSize: theme.typography.sizes.bodySmall,
    fontWeight: theme.typography.weights.bold,
    color: theme.colors.text.primary,
  },
  timelineChangeLog: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.secondary,
    lineHeight: 18,
    marginBottom: 4,
  },
  timelineTime: {
    fontSize: 11,
    color: theme.colors.text.tertiary,
  },
  // 底部操作栏
  actionBar: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.sm,
    paddingHorizontal: theme.spacing.md,
    paddingVertical: theme.spacing.sm,
    backgroundColor: theme.colors.background.primary,
    borderTopWidth: 1,
    borderTopColor: theme.colors.border.light,
    ...theme.layout.shadows.md,
  },
  actionIconBtn: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 8,
    paddingVertical: 4,
    minWidth: 52,
  },
  actionIconText: {
    fontSize: 11,
    color: theme.colors.text.secondary,
    marginTop: 2,
    fontWeight: theme.typography.weights.medium,
  },
  actionIconTextActive: {
    color: theme.colors.status.error,
  },
  actionPrimaryBtn: {
    flex: 1,
    borderRadius: theme.layout.borderRadius.md,
    overflow: 'hidden',
  },
  actionPrimaryGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    gap: 6,
  },
  actionPrimaryText: {
    fontSize: theme.typography.sizes.medium,
    fontWeight: theme.typography.weights.bold,
    color: '#fff',
  },
});