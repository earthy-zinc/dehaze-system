/**
 * 处理进度组件
 *
 * 显示当前进度条、处理阶段、已用时间、错误信息。
 */
import React from 'react';
import { View, Text, StyleSheet, ActivityIndicator } from 'react-native';
import { theme } from '@/theme';
import Icon from '@/components/Icon';
import type { TaskProgress } from '@/types/processing';
import { PROCESSING_STAGES } from '../services/processingApi';

interface ProcessingProgressProps {
  progress: TaskProgress;
}

const formatElapsed = (ms: number): string => {
  if (ms < 1000) return `${ms}ms`;
  const sec = Math.floor(ms / 1000);
  if (sec < 60) return `${sec}s`;
  const min = Math.floor(sec / 60);
  const remSec = sec % 60;
  return `${min}m${remSec}s`;
};

const ProcessingProgress: React.FC<ProcessingProgressProps> = ({ progress }) => {
  const { percent, stageLabel, elapsed, status, error } = progress;
  const isFailed = status === 'failed';
  const isCanceled = status === 'canceled';
  const isDone = status === 'success';

  const currentStageIndex = PROCESSING_STAGES.findIndex(s => s.key === status);
  return (
    <View style={styles.container}>
      {/* 顶部状态 */}
      <View style={styles.header}>
        <View style={styles.statusRow}>
          {isFailed || isCanceled ? (
            <Icon name="times" size={18} color={theme.colors.status.error} />
          ) : isDone ? (
            <Icon name="check-circle" size={18} color={theme.colors.status.success} />
          ) : (
            <ActivityIndicator size="small" color={theme.colors.primary} />
          )}
          <Text
            style={[
              styles.stageLabel,
              (isFailed || isCanceled) && styles.stageLabelError,
              isDone && styles.stageLabelSuccess,
            ]}
          >
            {isFailed ? '处理失败' : isCanceled ? '已取消' : isDone ? '处理完成' : stageLabel}
          </Text>
        </View>
        <Text style={styles.percentText}>{percent}%</Text>
      </View>

      {/* 进度条 */}
      <View style={styles.barContainer}>
        <View
          style={[
            styles.barFill,
            { width: `${percent}%` },
            isFailed && styles.barFillError,
            isCanceled && styles.barFillCanceled,
            isDone && styles.barFillSuccess,
          ]}
        />
      </View>

      {/* 阶段列表 */}
      <View style={styles.stageList}>
        {PROCESSING_STAGES.map((stage, index) => {
          const isPast = currentStageIndex > index || isDone;
          const isCurrent = currentStageIndex === index && !isDone && !isFailed && !isCanceled;
          return (
            <View key={stage.key} style={styles.stageItem}>
              <View
                style={[
                  styles.stageDot,
                  isPast && styles.stageDotDone,
                  isCurrent && styles.stageDotActive,
                ]}
              >
                {isPast ? (
                  <Icon name="check-circle" size={10} color="#fff" />
                ) : isCurrent ? (
                  <ActivityIndicator size="small" color="#fff" />
                ) : (
                  <View style={styles.stageDotPending} />
                )}
              </View>
              <Text
                style={[
                  styles.stageText,
                  isPast && styles.stageTextDone,
                  isCurrent && styles.stageTextActive,
                ]}
              >
                {stage.label}
              </Text>
            </View>
          );
        })}
      </View>

      {/* 已用时间 */}
      {!isFailed && !isCanceled && (
        <View style={styles.metaRow}>
          <Icon name="clock" size={14} color={theme.colors.text.tertiary} />
          <Text style={styles.metaText}>已用时间：{formatElapsed(elapsed)}</Text>
        </View>
      )}

      {/* 错误信息 */}
      {(isFailed || isCanceled) && error && (
        <View style={styles.errorContainer}>
          <Icon name="times" size={14} color={theme.colors.status.error} />
          <Text style={styles.errorText}>{error}</Text>
        </View>
      )}
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
    marginBottom: theme.spacing.sm,
  },
  statusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.sm,
  },
  stageLabel: {
    fontSize: theme.typography.sizes.body,
    fontWeight: theme.typography.weights.semibold,
    color: theme.colors.text.primary,
  },
  stageLabelError: {
    color: theme.colors.status.error,
  },
  stageLabelSuccess: {
    color: theme.colors.status.success,
  },
  percentText: {
    fontSize: theme.typography.sizes.body,
    fontWeight: theme.typography.weights.bold,
    color: theme.colors.primary,
  },
  barContainer: {
    height: 8,
    backgroundColor: theme.colors.background.tertiary,
    borderRadius: theme.layout.borderRadius.full,
    overflow: 'hidden',
    marginBottom: theme.spacing.md,
  },
  barFill: {
    height: '100%',
    backgroundColor: theme.colors.primary,
    borderRadius: theme.layout.borderRadius.full,
  },
  barFillError: {
    backgroundColor: theme.colors.status.error,
  },
  barFillCanceled: {
    backgroundColor: theme.colors.text.tertiary,
  },
  barFillSuccess: {
    backgroundColor: theme.colors.status.success,
  },
  stageList: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: theme.spacing.sm,
    marginBottom: theme.spacing.md,
  },
  stageItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    minWidth: 80,
  },
  stageDot: {
    width: 14,
    height: 14,
    borderRadius: 7,
    backgroundColor: theme.colors.background.tertiary,
    justifyContent: 'center',
    alignItems: 'center',
  },
  stageDotDone: {
    backgroundColor: theme.colors.status.success,
  },
  stageDotActive: {
    backgroundColor: theme.colors.primary,
  },
  stageDotPending: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: theme.colors.text.tertiary,
  },
  stageText: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.tertiary,
  },
  stageTextDone: {
    color: theme.colors.text.secondary,
  },
  stageTextActive: {
    color: theme.colors.primary,
    fontWeight: theme.typography.weights.medium,
  },
  metaRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  metaText: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.tertiary,
  },
  errorContainer: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 4,
    backgroundColor: `${theme.colors.status.error}10`,
    padding: theme.spacing.sm,
    borderRadius: theme.layout.borderRadius.sm,
    marginTop: theme.spacing.xs,
  },
  errorText: {
    flex: 1,
    fontSize: theme.typography.sizes.small,
    color: theme.colors.status.error,
    lineHeight: 18,
  },
});

export default ProcessingProgress;
