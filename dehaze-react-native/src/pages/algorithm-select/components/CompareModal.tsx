/**
 * 算法对比弹窗组件
 *
 * 调用 SDK AlgorithmAPI.compare（Java 后端 /api/v1/algorithms/select/compare）
 * 获取多算法对比，展示处理耗时与量化指标（PSNR/SSIM 等），并支持选择其一进入处理流程。
 */

import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Modal,
  ScrollView,
  ActivityIndicator,
  TouchableOpacity,
} from 'react-native';
import Icon from '@/components/Icon';
import { theme } from '@/theme';
import { AlgorithmAPI } from 'dehaze-sdk-js';
import type { Algorithm, AlgorithmCompareVO } from 'dehaze-sdk-js';

interface CompareModalProps {
  visible: boolean;
  algorithms: Algorithm[];
  /** 待对比图片 URL（可选，传给后端用于实际效果对比） */
  imageUrl?: string;
  onClose: () => void;
  onSelect: (algorithm: Algorithm) => void;
}

const CompareModal: React.FC<CompareModalProps> = ({
  visible,
  algorithms,
  imageUrl,
  onClose,
  onSelect,
}) => {
  const [loading, setLoading] = useState(true);
  const [results, setResults] = useState<AlgorithmCompareVO[]>([]);
  const [error, setError] = useState<string | null>(null);

  /** 解析后端返回的指标 JSON（如 {"psnr":38.2,"ssim":0.98}），输出可读字符串 */
  const formatMetrics = (metrics?: string): string | undefined => {
    if (!metrics) return undefined;
    try {
      const m = JSON.parse(metrics) as Record<string, number | string>;
      return Object.entries(m)
        .filter(([, v]) => v !== undefined && v !== null)
        .map(([k, v]) => `${k.toUpperCase()} ${v}`)
        .join(' · ');
    } catch {
      return metrics;
    }
  };

  const renderMetricRow = (
    label: string,
    getValue: (r: AlgorithmCompareVO) => string | undefined,
  ) => (
    <View style={styles.metricRow}>
      <Text style={styles.metricLabel}>{label}</Text>
      {results.map(r => (
        <Text key={r.algorithmId} style={styles.metricValue}>
          {getValue(r) ?? '-'}
        </Text>
      ))}
    </View>
  );

  useEffect(() => {
    if (!visible || algorithms.length < 2) return;

    let cancelled = false;
    setLoading(true);
    setError(null);

    AlgorithmAPI.compare({ algorithmIds: algorithms.map(a => a.id), imageUrl })
      .then(data => {
        if (!cancelled) {
          setResults(data || []);
          setLoading(false);
        }
      })
      .catch(err => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : '对比失败');
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [visible, algorithms, imageUrl]);

  /** 根据对比结果 ID 找回完整的 Algorithm（用于「使用」按钮跳转） */
  const findAlgorithm = (id: number): Algorithm | undefined =>
    algorithms.find(a => a.id === id);

  return (
    <Modal visible={visible} animationType="slide" transparent={false}>
      <View style={styles.container}>
        {/* 头部 */}
        <View style={styles.header}>
          <Text style={styles.title}>算法对比</Text>
          <TouchableOpacity onPress={onClose} style={styles.closeButton}>
            <Icon name="close" size={24} color={theme.colors.text.primary} />
          </TouchableOpacity>
        </View>

        {loading ? (
          <View style={styles.centerContainer}>
            <ActivityIndicator size="large" color={theme.colors.primary} />
            <Text style={styles.loadingText}>正在生成对比数据...</Text>
          </View>
        ) : error ? (
          <View style={styles.centerContainer}>
            <Icon name="alert-circle" size={48} color={theme.colors.status.error} />
            <Text style={styles.errorText}>{error}</Text>
          </View>
        ) : (
          <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
            {/* 算法名称行 */}
            <View style={styles.metricRow}>
              <Text style={styles.metricLabel}>算法</Text>
              {results.map(r => (
                <Text key={r.algorithmId} style={styles.algorithmName} numberOfLines={2}>
                  {r.algorithmName}
                </Text>
              ))}
            </View>

            {/* 处理耗时 */}
            {renderMetricRow('耗时(ms)', r =>
              r.time != null ? String(r.time) : undefined,
            )}

            {/* 评估指标 */}
            {renderMetricRow('评估指标', r => formatMetrics(r.metrics))}

            {/* 操作按钮 */}
            <View style={styles.actionRow}>
              {results.map(r => {
                const algorithm = findAlgorithm(r.algorithmId);
                if (!algorithm) return null;
                return (
                  <TouchableOpacity
                    key={r.algorithmId}
                    style={styles.useButton}
                    onPress={() => {
                      onSelect(algorithm);
                      onClose();
                    }}
                  >
                    <Text style={styles.useButtonText}>使用 {r.algorithmName}</Text>
                  </TouchableOpacity>
                );
              })}
            </View>
          </ScrollView>
        )}
      </View>
    </Modal>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: theme.colors.background.secondary,
    paddingTop: 60,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: theme.spacing.lg,
    paddingBottom: theme.spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: theme.colors.border.light,
  },
  title: {
    fontSize: theme.typography.sizes.h5,
    fontWeight: theme.typography.weights.bold,
    color: theme.colors.text.primary,
  },
  closeButton: {
    padding: theme.spacing.xs,
  },
  content: {
    flex: 1,
    padding: theme.spacing.lg,
  },
  centerContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: theme.spacing.md,
  },
  loadingText: {
    fontSize: theme.typography.sizes.medium,
    color: theme.colors.text.secondary,
  },
  errorText: {
    fontSize: theme.typography.sizes.medium,
    color: theme.colors.status.error,
  },
  metricRow: {
    flexDirection: 'row',
    paddingVertical: theme.spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: theme.colors.border.light,
  },
  metricLabel: {
    width: 80,
    fontSize: theme.typography.sizes.medium,
    fontWeight: theme.typography.weights.medium,
    color: theme.colors.text.secondary,
  },
  metricValue: {
    flex: 1,
    fontSize: theme.typography.sizes.medium,
    color: theme.colors.text.primary,
    textAlign: 'center',
  },
  algorithmName: {
    flex: 1,
    fontSize: theme.typography.sizes.medium,
    fontWeight: theme.typography.weights.semibold,
    color: theme.colors.primary,
    textAlign: 'center',
  },
  actionRow: {
    flexDirection: 'row',
    gap: theme.spacing.sm,
    marginTop: theme.spacing.lg,
  },
  useButton: {
    flex: 1,
    backgroundColor: theme.colors.primary,
    paddingVertical: theme.spacing.md,
    borderRadius: theme.layout.borderRadius.md,
    alignItems: 'center',
  },
  useButtonText: {
    fontSize: theme.typography.sizes.medium,
    fontWeight: theme.typography.weights.semibold,
    color: '#fff',
  },
});

export default CompareModal;
