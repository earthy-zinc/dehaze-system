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
import { formatDuration } from '@/utils/time';
import type { TaskProgress, TaskStatus } from '@/types/processing';

interface ProcessingProgressProps {
  progress: TaskProgress;
}

/** 状态显示配置（避免多处嵌套三元与重复状态判断） */
const STATUS_CONFIG: Record<
  TaskStatus,
  { label: string; isError: boolean; isSuccess: boolean; showSpinner: boolean }
> = {
  idle: { label: '去雾处理中', isError: false, isSuccess: false, showSpinner: true },
  processing: { label: '去雾处理中', isError: false, isSuccess: false, showSpinner: true },
  success: { label: '处理完成', isError: false, isSuccess: true, showSpinner: false },
  failed: { label: '处理失败', isError: true, isSuccess: false, showSpinner: false },
  canceled: { label: '已取消', isError: true, isSuccess: false, showSpinner: false },
};

const ProcessingProgress: React.FC<ProcessingProgressProps> = ({ progress }) => {
  const { status, elapsed, error } = progress;
  const config = STATUS_CONFIG[status];
  const showElapsed = config.showSpinner || config.isSuccess;

  return (
    <View style={styles.container}>
      {/* 顶部状态 */}
      <View style={styles.header}>
        <View style={styles.statusRow}>
          {config.isError ? (
            <Icon name="times" size={18} color={theme.colors.status.error} />
          ) : config.isSuccess ? (
            <Icon name="check-circle" size={18} color={theme.colors.status.success} />
          ) : (
            <ActivityIndicator size="small" color={theme.colors.primary} />
          )}
          <Text
            style={[
              styles.stageLabel,
              config.isError && styles.stageLabelError,
              config.isSuccess && styles.stageLabelSuccess,
            ]}
          >
            {config.label}
          </Text>
        </View>
      </View>

      {/* 已用时间（处理中或完成时展示）*/}
      {showElapsed && (
        <View style={styles.metaRow}>
          <Icon name="clock" size={14} color={theme.colors.text.tertiary} />
          <Text style={styles.metaText}>已用时间：{formatDuration(elapsed)}</Text>
        </View>
      )}

      {/* 错误信息 */}
      {config.isError && error && (
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
