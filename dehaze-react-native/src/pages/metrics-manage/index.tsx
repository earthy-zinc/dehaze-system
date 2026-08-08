/**
 * 指标管理页面（L2，工具 Tab 入口）
 *
 * 对应 05-菜单与页面层级规划 2.2 工具 → 指标管理：
 * - ModelAPI.getEvalMetrics 查询评估指标历史（已完成任务）
 * - ModelAPI.getEvalLogs 评估日志列表
 * - 指标查询/筛选/对比（选择多条记录对比 PSNR/SSIM/LPIPS）
 * - 与 L3 compare/Metrics 的区别：L2 管理页（列表+筛选+对比表格），L3 是沉浸对比时的指标叠加
 */
import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  View,
  Text,
  FlatList,
  StyleSheet,
  TouchableOpacity,
  TextInput,
  ScrollView,
  RefreshControl,
  ActivityIndicator,
  Alert,
} from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { ToolsStackParamList } from '@/routes/types';
import { AppHeader } from '@/layout';
import { theme } from '@/theme';
import Icon from '@/components/Icon';
import Card from '@/components/Card';
import Badge from '@/components/Badge';
import EmptyState from '@/components/EmptyState';
import { ModelAPI } from 'dehaze-sdk-js';
import type { EvalMetricsVO, EvalLogVO } from 'dehaze-sdk-js';

type Props = NativeStackScreenProps<ToolsStackParamList, 'MetricsManage'>;

type TabKey = 'metrics' | 'logs';

interface MetricFieldDef {
  key: string;
  label: string;
  unit?: string;
  better?: 'higher' | 'lower';
}

const METRIC_FIELDS: MetricFieldDef[] = [
  { key: 'psnr', label: 'PSNR', unit: 'dB', better: 'higher' },
  { key: 'ssim', label: 'SSIM', better: 'higher' },
  { key: 'mse', label: 'MSE', better: 'lower' },
  { key: 'lpips', label: 'LPIPS', better: 'lower' },
  { key: 'entropy', label: '信息熵' },
  { key: 'niqe', label: 'NIQE', better: 'lower' },
];

const MetricsManageScreen: React.FC<Props> = ({ navigation }) => {
  const [activeTab, setActiveTab] = useState<TabKey>('metrics');

  const [metricsList, setMetricsList] = useState<EvalMetricsVO[]>([]);
  const [metricsLoading, setMetricsLoading] = useState(true);
  const [metricsRefreshing, setMetricsRefreshing] = useState(false);

  const [logsList, setLogsList] = useState<EvalLogVO[]>([]);
  const [logsLoading, setLogsLoading] = useState(true);
  const [logsRefreshing, setLogsRefreshing] = useState(false);

  const [searchKeyword, setSearchKeyword] = useState('');
  const [compareIds, setCompareIds] = useState<Set<number>>(new Set());

  const loadMetrics = useCallback(async (isRefresh = false) => {
    try {
      if (isRefresh) setMetricsRefreshing(true);
      else setMetricsLoading(true);
      const result = await ModelAPI.getEvalMetrics({ pageNum: 1, pageSize: 100 });
      setMetricsList(result?.list || []);
    } catch {
      /* 静默失败 */
    } finally {
      setMetricsLoading(false);
      setMetricsRefreshing(false);
    }
  }, []);

  const loadLogs = useCallback(async (isRefresh = false) => {
    try {
      if (isRefresh) setLogsRefreshing(true);
      else setLogsLoading(true);
      const result = await ModelAPI.getEvalLogs({ pageNum: 1, pageSize: 100 });
      setLogsList(result?.list || []);
    } catch {
      /* 静默失败 */
    } finally {
      setLogsLoading(false);
      setLogsRefreshing(false);
    }
  }, []);

  useEffect(() => {
    loadMetrics();
    loadLogs();
  }, [loadMetrics, loadLogs]);

  const filteredMetrics = useMemo(() => {
    const kw = searchKeyword.trim().toLowerCase();
    if (!kw) return metricsList;
    return metricsList.filter(m => (m.algorithmName || '').toLowerCase().includes(kw));
  }, [metricsList, searchKeyword]);

  const filteredLogs = useMemo(() => {
    const kw = searchKeyword.trim().toLowerCase();
    if (!kw) return logsList;
    return logsList.filter(l => (l.algorithmName || '').toLowerCase().includes(kw));
  }, [logsList, searchKeyword]);

  const handleToggleCompare = useCallback((id: number) => {
    setCompareIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) { next.delete(id); }
      else {
        if (next.size >= 4) { Alert.alert('提示', '最多选择 4 条记录进行对比'); return prev; }
        next.add(id);
      }
      return next;
    });
  }, []);

  const formatMetric = (value: number | undefined, unit?: string): string => {
    if (value === undefined || value === null) return '—';
    const formatted = value < 1 ? value.toFixed(4) : value.toFixed(2);
    return unit ? `${formatted} ${unit}` : formatted;
  };

  const renderMetricsItem = useCallback(
    ({ item }: { item: EvalMetricsVO }) => {
      const isSelected = compareIds.has(item.id);
      const metrics = item.metrics || {};

      return (
        <TouchableOpacity
          style={[styles.listItem, isSelected && styles.listItemSelected]}
          onPress={() => handleToggleCompare(item.id)}
          activeOpacity={0.7}
        >
          <View style={styles.listItemHeader}>
            <View style={styles.listItemInfo}>
              <Text style={styles.listItemTitle} numberOfLines={1}>
                {item.algorithmName || `算法 #${item.algorithmId}`}
              </Text>
              <Text style={styles.listItemTime}>{item.createTime || '—'}</Text>
            </View>
            {isSelected && (
              <Icon name="success" size={18} color={theme.colors.primary} />
            )}
          </View>
          <View style={styles.metricsRow}>
            {METRIC_FIELDS.slice(0, 4).map(field => {
              const val = metrics[field.key];
              return (
                <View key={field.key} style={styles.metricMini}>
                  <Text style={styles.metricMiniLabel}>{field.label}</Text>
                  <Text style={[styles.metricMiniValue, typeof val === 'number' && getMetricStyle(field, val)]}>
                    {formatMetric(val, field.unit)}
                  </Text>
                </View>
              );
            })}
          </View>
          {item.time != null && <Text style={styles.listItemTime}>耗时：{item.time}ms</Text>}
        </TouchableOpacity>
      );
    },
    [compareIds, handleToggleCompare],
  );

  const renderLogsItem = useCallback(
    ({ item }: { item: EvalLogVO }) => {
      const statusLabel = item.status === 2 ? '已完成' : item.status === 3 ? '失败' : '处理中';
      const statusVariant = item.status === 2 ? 'success' as const : 'secondary' as const;
      return (
        <View style={styles.listItem}>
          <View style={styles.listItemHeader}>
            <View style={styles.listItemInfo}>
              <Text style={styles.listItemTitle} numberOfLines={1}>
                {item.algorithmName || `算法 #${item.algorithmId}`}
              </Text>
              <Text style={styles.listItemTime}>{item.createTime || '—'}</Text>
            </View>
            <Badge text={statusLabel} variant={statusVariant} size="small" />
          </View>
          {item.errorMessage && <Text style={styles.errorText} numberOfLines={1}>{item.errorMessage}</Text>}
          {item.time != null && <Text style={styles.listItemTime}>耗时：{item.time}ms</Text>}
        </View>
      );
    },
    [],
  );

  const selectedMetrics = metricsList.filter(m => compareIds.has(m.id));

  const tabs: { key: TabKey; label: string }[] = [
    { key: 'metrics', label: '指标历史' },
    { key: 'logs', label: '评估日志' },
  ];

  return (
    <View style={styles.container}>
      <AppHeader title="指标管理" showBack onBackPress={() => navigation.goBack()} />
      {/* Tab */}
      <View style={styles.tabContainer}>
        {tabs.map(tab => {
          const isActive = activeTab === tab.key;
          return (
            <TouchableOpacity
              key={tab.key}
              style={[styles.tab, isActive && styles.tabActive]}
              onPress={() => setActiveTab(tab.key)}
              activeOpacity={0.7}
            >
              <Text style={[styles.tabText, isActive && styles.tabTextActive]}>{tab.label}</Text>
            </TouchableOpacity>
          );
        })}
      </View>

      {/* 搜索栏 */}
      <View style={styles.searchContainer}>
        <Icon name="search" size={18} color={theme.colors.text.tertiary} />
        <TextInput
          style={styles.searchInput}
          placeholder="搜索算法名称..."
          placeholderTextColor={theme.colors.text.tertiary}
          value={searchKeyword}
          onChangeText={setSearchKeyword}
          autoCorrect={false}
          autoCapitalize="none"
        />
        {searchKeyword.length > 0 && (
          <TouchableOpacity onPress={() => setSearchKeyword('')} hitSlop={8}>
            <Icon name="cancel" size={18} color={theme.colors.text.tertiary} />
          </TouchableOpacity>
        )}
      </View>

      {/* 对比表格 */}
      {activeTab === 'metrics' && selectedMetrics.length >= 2 && (
        <Card style={styles.compareCard}>
          <View style={styles.compareHeader}>
            <Text style={styles.compareTitle}>指标对比（{selectedMetrics.length} 条）</Text>
            <TouchableOpacity onPress={() => setCompareIds(new Set())}>
              <Text style={styles.clearCompareText}>清空</Text>
            </TouchableOpacity>
          </View>
          <ScrollView horizontal showsHorizontalScrollIndicator={false}>
            <View>
              <View style={styles.compareRow}>
                <View style={styles.compareLabelCell}>
                  <Text style={styles.compareLabelText}>算法</Text>
                </View>
                {selectedMetrics.map(m => (
                  <View key={m.id} style={styles.compareValueCell}>
                    <Text style={styles.compareHeaderText} numberOfLines={1}>
                      {m.algorithmName || `#${m.algorithmId}`}
                    </Text>
                  </View>
                ))}
              </View>
              {METRIC_FIELDS.map(field => (
                <View key={field.key} style={styles.compareRow}>
                  <View style={styles.compareLabelCell}>
                    <Text style={styles.compareLabelText}>{field.label}</Text>
                  </View>
                  {selectedMetrics.map(m => {
                    const val = m.metrics?.[field.key];
                    const isBest = field.better && selectedMetrics.length > 1
                      ? field.better === 'higher'
                        ? val === Math.max(...selectedMetrics.map(x => x.metrics?.[field.key] ?? -Infinity))
                        : val === Math.min(...selectedMetrics.map(x => x.metrics?.[field.key] ?? Infinity))
                      : false;
                    return (
                      <View key={m.id} style={styles.compareValueCell}>
                        <Text style={[styles.compareValueText, isBest && styles.compareValueBest]}>
                          {formatMetric(val, field.unit)}
                        </Text>
                      </View>
                    );
                  })}
                </View>
              ))}
            </View>
          </ScrollView>
        </Card>
      )}

      {/* 列表 */}
      {activeTab === 'metrics' ? (
        <FlatList
          data={filteredMetrics}
          renderItem={renderMetricsItem}
          keyExtractor={item => item.id.toString()}
          contentContainerStyle={styles.listContent}
          showsVerticalScrollIndicator={false}
          refreshControl={
            <RefreshControl refreshing={metricsRefreshing} onRefresh={() => loadMetrics(true)} tintColor={theme.colors.primary} colors={[theme.colors.primary]} />
          }
          ListEmptyComponent={
            metricsLoading ? (
              <View style={styles.loadingContainer}>
                <ActivityIndicator size="large" color={theme.colors.primary} />
                <Text style={styles.loadingText}>加载指标历史...</Text>
              </View>
            ) : (
              <EmptyState icon="chart-line" title="暂无指标记录" description="完成去雾处理后，评估结果将出现在这里" />
            )
          }
        />
      ) : (
        <FlatList
          data={filteredLogs}
          renderItem={renderLogsItem}
          keyExtractor={item => item.id.toString()}
          contentContainerStyle={styles.listContent}
          showsVerticalScrollIndicator={false}
          refreshControl={
            <RefreshControl refreshing={logsRefreshing} onRefresh={() => loadLogs(true)} tintColor={theme.colors.primary} colors={[theme.colors.primary]} />
          }
          ListEmptyComponent={
            logsLoading ? (
              <View style={styles.loadingContainer}>
                <ActivityIndicator size="large" color={theme.colors.primary} />
                <Text style={styles.loadingText}>加载评估日志...</Text>
              </View>
            ) : (
              <EmptyState icon="clipboard" title="暂无评估日志" description="评估任务记录将出现在这里" />
            )
          }
        />
      )}
    </View>
  );
};

function getMetricStyle(field: MetricFieldDef, val: number) {
  if (!field.better) return null;
  if (field.better === 'higher') {
    if (field.key === 'psnr') return val > 25 ? styles.metricGood : styles.metricBad;
    if (field.key === 'ssim') return val > 0.85 ? styles.metricGood : styles.metricBad;
  }
  if (field.better === 'lower') {
    return val < 100 ? styles.metricGood : styles.metricBad;
  }
  return null;
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: theme.colors.background.secondary },
  tabContainer: {
    flexDirection: 'row', backgroundColor: theme.colors.background.primary,
    paddingHorizontal: theme.spacing.md, paddingTop: theme.spacing.md, paddingBottom: theme.spacing.sm,
    gap: theme.spacing.xs, borderBottomWidth: 1, borderBottomColor: theme.colors.border.light,
  },
  tab: { flex: 1, paddingVertical: theme.spacing.sm, borderRadius: theme.layout.borderRadius.md, alignItems: 'center', backgroundColor: theme.colors.background.tertiary },
  tabActive: { backgroundColor: theme.colors.primary },
  tabText: { fontSize: theme.typography.sizes.medium, fontWeight: theme.typography.weights.medium, color: theme.colors.text.secondary },
  tabTextActive: { color: '#fff', fontWeight: theme.typography.weights.semibold },
  searchContainer: {
    flexDirection: 'row', alignItems: 'center', backgroundColor: theme.colors.background.primary,
    marginHorizontal: theme.spacing.md, marginTop: theme.spacing.sm, marginBottom: theme.spacing.sm,
    borderRadius: theme.layout.borderRadius.md, paddingHorizontal: theme.spacing.md, paddingVertical: theme.spacing.sm,
    gap: theme.spacing.sm, ...theme.layout.shadows.sm,
  },
  searchInput: { flex: 1, fontSize: theme.typography.sizes.medium, color: theme.colors.text.primary, padding: 0 },
  listContent: { paddingHorizontal: theme.spacing.md, paddingBottom: theme.spacing.xxxl },
  loadingContainer: { alignItems: 'center', paddingVertical: theme.spacing.xxxl, gap: theme.spacing.sm },
  loadingText: { fontSize: theme.typography.sizes.medium, color: theme.colors.text.secondary },
  listItem: { backgroundColor: theme.colors.background.primary, borderRadius: theme.layout.borderRadius.md, padding: theme.spacing.md, marginBottom: theme.spacing.sm, ...theme.layout.shadows.sm },
  listItemSelected: { borderWidth: 2, borderColor: theme.colors.primary },
  listItemHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start' },
  listItemInfo: { flex: 1, gap: 2 },
  listItemTitle: { fontSize: theme.typography.sizes.medium, fontWeight: theme.typography.weights.semibold, color: theme.colors.text.primary },
  listItemTime: { fontSize: theme.typography.sizes.small, color: theme.colors.text.tertiary },
  metricsRow: { flexDirection: 'row', marginTop: theme.spacing.sm, paddingTop: theme.spacing.sm, borderTopWidth: StyleSheet.hairlineWidth, borderTopColor: theme.colors.border.light, gap: 4 },
  metricMini: { flex: 1, alignItems: 'center', paddingHorizontal: 4 },
  metricMiniLabel: { fontSize: 9, color: theme.colors.text.tertiary, marginBottom: 2 },
  metricMiniValue: { fontSize: 12, fontWeight: theme.typography.weights.bold, color: theme.colors.text.primary },
  metricGood: { color: theme.colors.status.success },
  metricBad: { color: theme.colors.status.error },
  errorText: { fontSize: theme.typography.sizes.small, color: theme.colors.status.error, marginTop: 4 },
  compareCard: { marginHorizontal: theme.spacing.md, marginBottom: theme.spacing.sm },
  compareHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: theme.spacing.md },
  compareTitle: { fontSize: theme.typography.sizes.medium, fontWeight: theme.typography.weights.semibold, color: theme.colors.text.primary },
  clearCompareText: { fontSize: theme.typography.sizes.small, color: theme.colors.primary },
  compareRow: { flexDirection: 'row', borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: theme.colors.border.light },
  compareLabelCell: { width: 70, paddingVertical: theme.spacing.sm, paddingRight: theme.spacing.sm },
  compareLabelText: { fontSize: theme.typography.sizes.small, fontWeight: theme.typography.weights.medium, color: theme.colors.text.secondary },
  compareValueCell: { width: 90, paddingVertical: theme.spacing.sm, alignItems: 'center' },
  compareHeaderText: { fontSize: theme.typography.sizes.small, fontWeight: theme.typography.weights.semibold, color: theme.colors.text.primary },
  compareValueText: { fontSize: theme.typography.sizes.small, color: theme.colors.text.primary, fontWeight: theme.typography.weights.medium },
  compareValueBest: { color: theme.colors.status.success, fontWeight: theme.typography.weights.bold },
});

export default MetricsManageScreen;
