/**
 * 算法对比弹窗组件
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
import type { Algorithm, CompareResult } from '@/types/algorithm';
import AlgorithmSelectAPI from '@/api/algorithm-select';

interface CompareModalProps {
  visible: boolean;
  algorithms: Algorithm[];
  onClose: () => void;
  onSelect: (algorithm: Algorithm) => void;
}

const CompareModal: React.FC<CompareModalProps> = ({
  visible,
  algorithms,
  onClose,
  onSelect,
}) => {
  const [loading, setLoading] = useState(true);
  const [results, setResults] = useState<CompareResult[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!visible || algorithms.length < 2) return;

    let cancelled = false;
    setLoading(true);
    setError(null);

    AlgorithmSelectAPI.compare(algorithms.map(a => a.id))
      .then(data => {
        if (!cancelled) {
          setResults(data);
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
  }, [visible, algorithms]);

  const renderMetricRow = (label: string, getValue: (r: CompareResult) => string | undefined) => (
    <View style={styles.metricRow}>
      <Text style={styles.metricLabel}>{label}</Text>
      {results.map(r => (
        <Text key={r.algorithm.id} style={styles.metricValue}>
          {getValue(r) ?? '-'}
        </Text>
      ))}
    </View>
  );

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
                <Text key={r.algorithm.id} style={styles.algorithmName} numberOfLines={2}>
                  {r.algorithm.name}
                </Text>
              ))}
            </View>

            {/* 类型行 */}
            <View style={styles.metricRow}>
              <Text style={styles.metricLabel}>类型</Text>
              {results.map(r => (
                <Text key={r.algorithm.id} style={styles.metricValue}>
                  {r.algorithm.type || '-'}
                </Text>
              ))}
            </View>

            {/* PSNR */}
            {renderMetricRow('PSNR', r => r.metrics?.psnr?.toFixed(2))}

            {/* SSIM */}
            {renderMetricRow('SSIM', r => r.metrics?.ssim?.toFixed(4))}

            {/* 速度 */}
            {renderMetricRow('速度(ms)', r => r.metrics?.speed?.toFixed(0))}

            {/* 评分 */}
            {renderMetricRow('评分', r => r.metrics?.rating?.toFixed(1))}

            {/* 版本 */}
            <View style={styles.metricRow}>
              <Text style={styles.metricLabel}>版本</Text>
              {results.map(r => (
                <Text key={r.algorithm.id} style={styles.metricValue}>
                  {r.algorithm.version || '-'}
                </Text>
              ))}
            </View>

            {/* 大小 */}
            <View style={styles.metricRow}>
              <Text style={styles.metricLabel}>大小</Text>
              {results.map(r => (
                <Text key={r.algorithm.id} style={styles.metricValue}>
                  {r.algorithm.size || '-'}
                </Text>
              ))}
            </View>

            {/* 操作按钮 */}
            <View style={styles.actionRow}>
              {results.map(r => (
                <TouchableOpacity
                  key={r.algorithm.id}
                  style={styles.useButton}
                  onPress={() => {
                    onSelect(r.algorithm);
                    onClose();
                  }}
                >
                  <Text style={styles.useButtonText}>使用 {r.algorithm.name}</Text>
                </TouchableOpacity>
              ))}
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
    fontSize: theme.typography.sizes.body,
    color: theme.colors.text.secondary,
  },
  errorText: {
    fontSize: theme.typography.sizes.body,
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
    fontSize: theme.typography.sizes.body,
    fontWeight: theme.typography.weights.medium,
    color: theme.colors.text.secondary,
  },
  metricValue: {
    flex: 1,
    fontSize: theme.typography.sizes.body,
    color: theme.colors.text.primary,
    textAlign: 'center',
  },
  algorithmName: {
    flex: 1,
    fontSize: theme.typography.sizes.body,
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
    fontSize: theme.typography.sizes.body,
    fontWeight: theme.typography.weights.semibold,
    color: '#fff',
  },
});

export default CompareModal;
