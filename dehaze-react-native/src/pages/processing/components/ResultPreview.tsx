/**
 * 处理结果预览组件
 *
 * 显示原图与处理后的图片并排预览，提供：
 * - 进入效果对比页面（5 种模式）
 * - 重新处理
 * - 处理耗时与缓存标识
 */
import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ScrollView } from 'react-native';
import ImageLoader from '@/components/ImageLoader';
import Icon from '@/components/Icon';
import { theme } from '@/theme';
import type { ProcessingResult } from '@/types/processing';

interface ResultPreviewProps {
  originalUrl: string;
  result: ProcessingResult;
  onEnterCompare: () => void;
  onReprocess: () => void;
}

const formatTime = (ms: number): string => {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
};

const ResultPreview: React.FC<ResultPreviewProps> = ({
  originalUrl,
  result,
  onEnterCompare,
  onReprocess,
}) => {
  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <View style={styles.titleRow}>
          <Icon name="check-circle" size={18} color={theme.colors.status.success} />
          <Text style={styles.title}>处理完成</Text>
        </View>
        {result.fromCache && (
          <View style={styles.cacheBadge}>
            <Icon name="bolt" size={12} color={theme.colors.status.warning} />
            <Text style={styles.cacheText}>命中缓存</Text>
          </View>
        )}
      </View>

      <View style={styles.metaRow}>
        <View style={styles.metaItem}>
          <Icon name="clock" size={12} color={theme.colors.text.tertiary} />
          <Text style={styles.metaText}>耗时 {formatTime(result.time)}</Text>
        </View>
      </View>

      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.previewRow}
      >
        {/* 原图 */}
        <View style={styles.previewCard}>
          <View style={styles.previewLabelRow}>
            <View style={[styles.previewLabelBadge, styles.previewLabelBadgeOriginal]}>
              <Text style={styles.previewLabelText}>原图</Text>
            </View>
          </View>
          <ImageLoader
            source={{ uri: originalUrl }}
            style={styles.previewImage}
            resizeMode="contain"
          />
        </View>

        {/* 处理后 */}
        <View style={styles.previewCard}>
          <View style={styles.previewLabelRow}>
            <View style={[styles.previewLabelBadge, styles.previewLabelBadgeResult]}>
              <Text style={styles.previewLabelText}>去雾后</Text>
            </View>
          </View>
          {result.resultThumbnailUrl || result.resultUrl ? (
            <ImageLoader
              source={{ uri: result.resultThumbnailUrl || result.resultUrl }}
              style={styles.previewImage}
              resizeMode="contain"
            />
          ) : (
            <View style={[styles.previewImage, styles.previewPlaceholder]}>
              <Icon name="image" size={32} color={theme.colors.text.tertiary} />
            </View>
          )}
        </View>
      </ScrollView>

      <View style={styles.actions}>
        <TouchableOpacity
          style={styles.primaryAction}
          onPress={onEnterCompare}
          activeOpacity={0.8}
        >
          <Icon name="columns" size={16} color="#fff" />
          <Text style={styles.primaryActionText}>进入效果对比</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.secondaryAction}
          onPress={onReprocess}
          activeOpacity={0.8}
        >
          <Icon name="refresh" size={16} color={theme.colors.primary} />
          <Text style={styles.secondaryActionText}>重新处理</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: theme.colors.background.primary,
    borderRadius: theme.layout.borderRadius.lg,
    padding: theme.spacing.lg,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: theme.spacing.xs,
  },
  titleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.sm,
  },
  title: {
    fontSize: theme.typography.sizes.bodyLarge,
    fontWeight: theme.typography.weights.bold,
    color: theme.colors.text.primary,
  },
  cacheBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: `${theme.colors.status.warning}15`,
    paddingHorizontal: theme.spacing.sm,
    paddingVertical: 2,
    borderRadius: theme.layout.borderRadius.full,
  },
  cacheText: {
    fontSize: theme.typography.sizes.tiny,
    color: theme.colors.status.warning,
    fontWeight: theme.typography.weights.medium,
  },
  metaRow: {
    flexDirection: 'row',
    gap: theme.spacing.md,
    marginBottom: theme.spacing.md,
  },
  metaItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  metaText: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.tertiary,
  },
  previewRow: {
    gap: theme.spacing.md,
    paddingBottom: theme.spacing.sm,
  },
  previewCard: {
    width: 240,
    backgroundColor: theme.colors.background.secondary,
    borderRadius: theme.layout.borderRadius.md,
    overflow: 'hidden',
  },
  previewLabelRow: {
    paddingHorizontal: theme.spacing.sm,
    paddingVertical: theme.spacing.xs,
    backgroundColor: theme.colors.background.tertiary,
  },
  previewLabelBadge: {
    alignSelf: 'flex-start',
    paddingHorizontal: theme.spacing.sm,
    paddingVertical: 2,
    borderRadius: theme.layout.borderRadius.full,
  },
  previewLabelBadgeOriginal: {
    backgroundColor: theme.colors.text.tertiary,
  },
  previewLabelBadgeResult: {
    backgroundColor: theme.colors.status.success,
  },
  previewLabelText: {
    fontSize: theme.typography.sizes.tiny,
    color: '#fff',
    fontWeight: theme.typography.weights.medium,
  },
  previewImage: {
    width: '100%',
    height: 200,
  },
  previewPlaceholder: {
    backgroundColor: theme.colors.background.tertiary,
    alignItems: 'center',
    justifyContent: 'center',
  },
  actions: {
    flexDirection: 'row',
    gap: theme.spacing.sm,
    marginTop: theme.spacing.md,
  },
  primaryAction: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: theme.spacing.xs,
    paddingVertical: theme.spacing.md,
    borderRadius: theme.layout.borderRadius.md,
    backgroundColor: theme.colors.primary,
  },
  primaryActionText: {
    fontSize: theme.typography.sizes.body,
    fontWeight: theme.typography.weights.semibold,
    color: '#fff',
  },
  secondaryAction: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: theme.spacing.xs,
    paddingVertical: theme.spacing.md,
    paddingHorizontal: theme.spacing.lg,
    borderRadius: theme.layout.borderRadius.md,
    backgroundColor: `${theme.colors.primary}15`,
  },
  secondaryActionText: {
    fontSize: theme.typography.sizes.body,
    fontWeight: theme.typography.weights.medium,
    color: theme.colors.primary,
  },
});

export default ResultPreview;
