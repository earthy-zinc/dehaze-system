/**
 * 处理进度组件
 *
 * API 同步返回处理结果，仅展示真实处理状态、已用时间、错误信息。
 * 不展示模拟百分比或阶段化进度条。
 */
import React from 'react';
import { View, Text, StyleSheet, ActivityIndicator } from 'react-native';
import { theme } from '@/theme';
import Icon from '@/components/Icon';
import type { TaskProgress } from '@/types/processing';

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
  const { status, elapsed, error } = progress;
  const isFailed = status === 'failed';
  const isCanceled = status === 'canceled';
  const isDone = status === 'success';
  const isProcessing = !isFailed && !isCanceled && !isDone;

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
            {isFailed ? '处理失败' : isCanceled ? '已取消' : isDone ? '处理完成' : '去雾处理中'}
          </Text>
        </View>
      </View>

      {/* 已用时间（处理中或完成时展示）*/}
      {(isProcessing || isDone) && (
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
