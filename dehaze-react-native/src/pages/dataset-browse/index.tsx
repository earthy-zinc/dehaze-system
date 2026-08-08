/**
 * 数据集浏览版（L2，工具 Tab 入口）
 *
 * 对应 05-菜单与页面层级规划 2.2 工具 → 数据集：
 * - 数据集列表（DatasetAPI 列表，只看公开/共享）
 * - 数据集详情（图片网格浏览，复用现有 DatasetDetailSection 组件）
 * - 无创建/编辑/删除（管理归 dev-admin 的 system/dataset）
 * - 「使用该数据集」可带入去雾流程
 */
import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  FlatList,
  StyleSheet,
  TouchableOpacity,
  RefreshControl,
  ActivityIndicator,
} from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { ToolsStackParamList } from '@/routes/types';
import { AppHeader } from '@/layout';
import { theme } from '@/theme';
import Icon from '@/components/Icon';
import Card from '@/components/Card';
import EmptyState from '@/components/EmptyState';
import { DatasetAPI } from 'dehaze-sdk-js';
import type { Dataset } from 'dehaze-sdk-js';

// 复用现有组件
import DatasetDetailSection from '@/pages/dataset/components/DatasetDetailSection';
import SearchBar from '@/pages/dataset/components/SearchBar';

type Props = NativeStackScreenProps<ToolsStackParamList, 'DatasetBrowse'>;

const DatasetBrowseScreen: React.FC<Props> = ({ navigation }) => {
  const [currentView, setCurrentView] = useState<'list' | 'detail'>('list');
  const [selectedDatasetId, setSelectedDatasetId] = useState<number | null>(null);
  const [selectedDatasetName, setSelectedDatasetName] = useState('');

  // 列表状态
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchValue, setSearchValue] = useState('');

  /** 加载数据集列表 */
  const loadDatasets = useCallback(async (isRefresh = false) => {
    try {
      if (isRefresh) setRefreshing(true);
      else setLoading(true);
      setError(null);

      const result = await DatasetAPI.getList({
        keyword: searchValue.trim() || undefined,
        pageNum: 1,
        pageSize: 50,
        status: '1',
      });

      setDatasets((result?.list || []) as Dataset[]);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : '加载数据集失败';
      setError(msg);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [searchValue]);

  useFocusEffect(
    useCallback(() => {
      loadDatasets();
    }, [loadDatasets]),
  );

  const handleDatasetPress = useCallback((dataset: Dataset) => {
    setSelectedDatasetId(dataset.id);
    setSelectedDatasetName(dataset.name);
    setCurrentView('detail');
  }, []);

  const handleBack = useCallback(() => {
    setCurrentView('list');
    setSelectedDatasetId(null);
    setSelectedDatasetName('');
    setSearchValue('');
  }, []);

  const handleUseDataset = useCallback(
    (_dataset: Dataset) => {
      navigation.navigate('ImageInput');
    },
    [navigation],
  );

  const renderDatasetCard = useCallback(
    ({ item }: { item: Dataset }) => {
      const stats = item.statistics;
      return (
        <Card style={styles.card}>
          <TouchableOpacity
            activeOpacity={0.7}
            onPress={() => handleDatasetPress(item)}
          >
            <View style={styles.cardHeader}>
              <View style={styles.cardIcon}>
                <Icon name="database" size={20} color={theme.colors.secondary} />
              </View>
              <View style={styles.cardTitleWrap}>
                <Text style={styles.cardTitle} numberOfLines={1}>
                  {item.name}
                </Text>
                <Text style={styles.cardType} numberOfLines={1}>
                  {item.type || '通用数据集'}
                </Text>
              </View>
            </View>

            {item.description ? (
              <Text style={styles.cardDesc} numberOfLines={2}>
                {item.description}
              </Text>
            ) : null}

            {stats && (
              <View style={styles.statsRow}>
                <View style={styles.statItem}>
                  <Text style={styles.statValue}>{stats.itemCount ?? '—'}</Text>
                  <Text style={styles.statLabel}>数据项</Text>
                </View>
                <View style={styles.statDivider} />
                <View style={styles.statItem}>
                  <Text style={styles.statValue}>{stats.fileCount ?? '—'}</Text>
                  <Text style={styles.statLabel}>文件</Text>
                </View>
                <View style={styles.statDivider} />
                <View style={styles.statItem}>
                  <Text style={styles.statValue}>{stats.annotatedCount ?? '—'}</Text>
                  <Text style={styles.statLabel}>已标注</Text>
                </View>
              </View>
            )}
          </TouchableOpacity>

          <View style={styles.cardActions}>
            <TouchableOpacity
              style={styles.detailBtn}
              onPress={() => handleDatasetPress(item)}
              activeOpacity={0.7}
            >
              <Text style={styles.detailBtnText}>浏览图片</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.useBtn}
              onPress={() => handleUseDataset(item)}
              activeOpacity={0.8}
            >
              <Icon name="image" size={14} color="#fff" />
              <Text style={styles.useBtnText}>使用该数据集</Text>
            </TouchableOpacity>
          </View>
        </Card>
      );
    },
    [handleDatasetPress, handleUseDataset],
  );

  const keyExtractor = useCallback((item: Dataset) => item.id.toString(), []);

  // 详情视图
  if (currentView === 'detail' && selectedDatasetId) {
    return (
      <View style={styles.container}>
        <AppHeader title={selectedDatasetName || '数据集详情'} showBack onBackPress={handleBack} />
        <TouchableOpacity
          style={styles.backToListBtn}
          onPress={handleBack}
          activeOpacity={0.7}
        >
          <Icon name="back" size={14} color={theme.colors.primary} />
          <Text style={styles.backToListText}>返回数据集列表</Text>
        </TouchableOpacity>
        <View style={styles.detailWrap}>
          <DatasetDetailSection datasetId={selectedDatasetId} onBack={handleBack} />
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <AppHeader title="数据集" showBack onBackPress={() => navigation.goBack()} />
      <View style={styles.searchContainer}>
        <SearchBar
          value={searchValue}
          onChangeText={setSearchValue}
          placeholder="搜索数据集..."
        />
      </View>

      <FlatList
        data={datasets}
        renderItem={renderDatasetCard}
        keyExtractor={keyExtractor}
        contentContainerStyle={styles.listContent}
        showsVerticalScrollIndicator={false}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={() => loadDatasets(true)}
            tintColor={theme.colors.secondary}
            colors={[theme.colors.secondary]}
          />
        }
        ListEmptyComponent={
          loading ? (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="large" color={theme.colors.secondary} />
              <Text style={styles.loadingText}>加载数据集...</Text>
            </View>
          ) : error ? (
            <EmptyState icon="cloud-offline" title="加载失败" description={error} />
          ) : (
            <EmptyState
              icon="database"
              title="暂无数据集"
              description={searchValue ? '未找到匹配的数据集' : '还没有公开的数据集'}
            />
          )
        }
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: theme.colors.background.secondary,
  },
  searchContainer: {
    backgroundColor: theme.colors.background.primary,
    paddingHorizontal: theme.spacing.md,
    paddingTop: theme.spacing.md,
    paddingBottom: theme.spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: theme.colors.background.tertiary,
  },
  backToListBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: theme.spacing.md,
    paddingVertical: theme.spacing.sm,
    backgroundColor: theme.colors.background.primary,
    borderBottomWidth: 1,
    borderBottomColor: theme.colors.border.light,
  },
  backToListText: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.primary,
    fontWeight: theme.typography.weights.medium,
  },
  detailWrap: {
    flex: 1,
  },
  listContent: {
    paddingHorizontal: theme.spacing.md,
    paddingVertical: theme.spacing.sm,
    paddingBottom: theme.spacing.xxxl,
  },
  loadingContainer: {
    alignItems: 'center',
    paddingVertical: theme.spacing.xxxl,
    gap: theme.spacing.sm,
  },
  loadingText: {
    fontSize: theme.typography.sizes.medium,
    color: theme.colors.text.secondary,
  },
  card: {
    marginBottom: theme.spacing.sm,
  },
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.sm,
  },
  cardIcon: {
    width: 40,
    height: 40,
    borderRadius: theme.layout.borderRadius.md,
    backgroundColor: `${theme.colors.secondary}15`,
    justifyContent: 'center',
    alignItems: 'center',
  },
  cardTitleWrap: {
    flex: 1,
    gap: 2,
  },
  cardTitle: {
    fontSize: theme.typography.sizes.medium,
    fontWeight: theme.typography.weights.semibold,
    color: theme.colors.text.primary,
  },
  cardType: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.secondary,
  },
  cardDesc: {
    fontSize: theme.typography.sizes.bodySmall,
    color: theme.colors.text.secondary,
    lineHeight: 20,
    marginTop: theme.spacing.sm,
  },
  statsRow: {
    flexDirection: 'row',
    backgroundColor: theme.colors.background.tertiary,
    borderRadius: theme.layout.borderRadius.sm,
    paddingVertical: theme.spacing.sm,
    marginTop: theme.spacing.sm,
  },
  statItem: {
    flex: 1,
    alignItems: 'center',
  },
  statValue: {
    fontSize: theme.typography.sizes.large,
    fontWeight: theme.typography.weights.bold,
    color: theme.colors.primary,
  },
  statLabel: {
    fontSize: theme.typography.sizes.tiny,
    color: theme.colors.text.tertiary,
    marginTop: 2,
  },
  statDivider: {
    width: 1,
    backgroundColor: theme.colors.border.light,
  },
  cardActions: {
    flexDirection: 'row',
    gap: theme.spacing.sm,
    marginTop: theme.spacing.md,
    paddingTop: theme.spacing.sm,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: theme.colors.border.light,
  },
  detailBtn: {
    flex: 1,
    paddingVertical: theme.spacing.sm,
    borderRadius: theme.layout.borderRadius.sm,
    borderWidth: 1,
    borderColor: theme.colors.border.light,
    alignItems: 'center',
  },
  detailBtnText: {
    fontSize: theme.typography.sizes.small,
    fontWeight: theme.typography.weights.medium,
    color: theme.colors.text.secondary,
  },
  useBtn: {
    flex: 1.5,
    flexDirection: 'row',
    paddingVertical: theme.spacing.sm,
    borderRadius: theme.layout.borderRadius.sm,
    backgroundColor: theme.colors.secondary,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 4,
  },
  useBtnText: {
    fontSize: theme.typography.sizes.small,
    fontWeight: theme.typography.weights.semibold,
    color: '#fff',
  },
});

export default DatasetBrowseScreen;
