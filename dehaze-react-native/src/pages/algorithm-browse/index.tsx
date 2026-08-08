/**
 * 算法库浏览版（L2，工具 Tab 入口）
 *
 * 对应 05-菜单与页面层级规划 2.2 工具 → 算法库：
 * - 算法列表（AlgorithmAPI.tree，仅已发布，卡片展示）
 * - 算法详情查看（导航到 Algorithm 详情页）
 * - 「使用该算法」按钮 → 带入去雾流程（跨 Stack 导航到 Processing）
 * - 无审计/上下架/删除（管理归 dev-admin 的 system/algorithm）
 */
import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  View,
  Text,
  FlatList,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
  TextInput,
  RefreshControl,
} from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { ToolsStackParamList } from '@/routes/types';
import { AppHeader } from '@/layout';
import { theme } from '@/theme';
import Icon from '@/components/Icon';
import Card from '@/components/Card';
import Badge from '@/components/Badge';
import EmptyState from '@/components/EmptyState';
import { AlgorithmAPI } from 'dehaze-sdk-js';
import type { Algorithm } from 'dehaze-sdk-js';

type Props = NativeStackScreenProps<ToolsStackParamList, 'AlgorithmBrowse'>;

const AlgorithmBrowseScreen: React.FC<Props> = ({ navigation }) => {
  const [algorithms, setAlgorithms] = useState<Algorithm[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchKeyword, setSearchKeyword] = useState('');

  /** 加载算法列表（仅已发布） */
  const loadAlgorithms = useCallback(async (isRefresh = false) => {
    try {
      if (isRefresh) setRefreshing(true);
      else setLoading(true);
      setError(null);

      const data = await AlgorithmAPI.tree();
      const flatten = (nodes: Algorithm[]): Algorithm[] => {
        const result: Algorithm[] = [];
        for (const node of nodes) {
          if (node.children && node.children.length > 0) {
            result.push(...flatten(node.children));
          } else {
            result.push(node);
          }
        }
        return result;
      };
      setAlgorithms(flatten((data || []) as unknown as Algorithm[]));
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : '加载算法列表失败';
      setError(msg);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    loadAlgorithms();
  }, [loadAlgorithms]);

  /** 过滤算法 */
  const filteredAlgorithms = useMemo(() => {
    const kw = searchKeyword.trim().toLowerCase();
    if (!kw) return algorithms;
    return algorithms.filter(a =>
      (a.name || '').toLowerCase().includes(kw) ||
      (a.type || '').toLowerCase().includes(kw) ||
      (a.description || '').toLowerCase().includes(kw),
    );
  }, [algorithms, searchKeyword]);

  /** 使用算法 → 带入去雾流程（跨 Stack） */
  const handleUseAlgorithm = useCallback(
    (algorithm: Algorithm) => {
      navigation.navigate('Processing', { algorithmId: algorithm.id });
    },
    [navigation],
  );

  /** 查看算法详情 */
  const handleViewDetail = useCallback(
    (algorithm: Algorithm) => {
      navigation.navigate('Algorithm', { algorithmId: algorithm.id });
    },
    [navigation],
  );

  /** 渲染算法卡片 */
  const renderAlgorithmCard = useCallback(
    ({ item }: { item: Algorithm }) => (
      <Card style={styles.card}>
        <TouchableOpacity
          activeOpacity={0.7}
          onPress={() => handleViewDetail(item)}
        >
          <View style={styles.cardHeader}>
            <View style={styles.cardIcon}>
              <Icon name="brain" size={20} color={theme.colors.primary} />
            </View>
            <View style={styles.cardTitleWrap}>
              <Text style={styles.cardTitle} numberOfLines={1}>
                {item.name}
              </Text>
              <Text style={styles.cardType} numberOfLines={1}>
                {item.type || '通用类型'}
              </Text>
            </View>
            {item.status === 4 && (
              <Badge text="已发布" variant="success" size="small" />
            )}
          </View>

          {item.description ? (
            <Text style={styles.cardDesc} numberOfLines={2}>
              {item.description}
            </Text>
          ) : null}

          <View style={styles.cardMetrics}>
            {item.version ? (
              <View style={styles.metricTag}>
                <Text style={styles.metricTagText}>v{item.version}</Text>
              </View>
            ) : null}
            {item.size ? (
              <View style={styles.metricTag}>
                <Icon name="file" size={10} color={theme.colors.text.tertiary} />
                <Text style={styles.metricTagText}>{item.size}</Text>
              </View>
            ) : null}
          </View>
        </TouchableOpacity>

        <View style={styles.cardActions}>
          <TouchableOpacity
            style={styles.detailBtn}
            onPress={() => handleViewDetail(item)}
            activeOpacity={0.7}
          >
            <Text style={styles.detailBtnText}>查看详情</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.useBtn}
            onPress={() => handleUseAlgorithm(item)}
            activeOpacity={0.8}
          >
            <Icon name="bolt" size={14} color="#fff" />
            <Text style={styles.useBtnText}>使用该算法</Text>
          </TouchableOpacity>
        </View>
      </Card>
    ),
    [handleViewDetail, handleUseAlgorithm],
  );

  const keyExtractor = useCallback((item: Algorithm) => item.id.toString(), []);

  return (
    <View style={styles.container}>
      <AppHeader title="算法库" showBack onBackPress={() => navigation.goBack()} />
      <View style={styles.content}>
        {/* 搜索栏 */}
        <View style={styles.searchContainer}>
          <Icon name="search" size={18} color={theme.colors.text.tertiary} />
          <TextInput
            style={styles.searchInput}
            placeholder="搜索算法名称、类型或描述"
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

        {/* 推荐提示条 */}
        {!searchKeyword && (
          <TouchableOpacity
            style={styles.recommendBanner}
            onPress={() => navigation.navigate('AlgorithmSelect')}
            activeOpacity={0.7}
          >
            <Icon name="magic" size={16} color={theme.colors.primary} />
            <Text style={styles.recommendText}>
              上传图片后可在算法选择页获取智能推荐
            </Text>
            <Icon name="chevron-right" size={14} color={theme.colors.text.tertiary} />
          </TouchableOpacity>
        )}

        {/* 算法列表 */}
        <FlatList
          data={filteredAlgorithms}
          renderItem={renderAlgorithmCard}
          keyExtractor={keyExtractor}
          contentContainerStyle={styles.listContent}
          showsVerticalScrollIndicator={false}
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={() => loadAlgorithms(true)}
              tintColor={theme.colors.primary}
              colors={[theme.colors.primary]}
            />
          }
          ListEmptyComponent={
            loading ? (
              <View style={styles.loadingContainer}>
                <ActivityIndicator size="large" color={theme.colors.primary} />
                <Text style={styles.loadingText}>加载算法库...</Text>
              </View>
            ) : error ? (
              <EmptyState
                icon="cloud-offline"
                title="加载失败"
                description={error}
              />
            ) : (
              <EmptyState
                icon="brain"
                title={searchKeyword ? '未找到匹配算法' : '暂无算法'}
                description={
                  searchKeyword
                    ? '尝试其他关键词'
                    : '算法库暂时为空，请稍后再来'
                }
              />
            )
          }
        />
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: theme.colors.background.secondary,
  },
  content: {
    flex: 1,
  },
  searchContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: theme.colors.background.primary,
    marginHorizontal: theme.spacing.md,
    marginTop: theme.spacing.md,
    marginBottom: theme.spacing.sm,
    borderRadius: theme.layout.borderRadius.md,
    paddingHorizontal: theme.spacing.md,
    paddingVertical: theme.spacing.sm,
    gap: theme.spacing.sm,
    ...theme.layout.shadows.sm,
  },
  searchInput: {
    flex: 1,
    fontSize: theme.typography.sizes.medium,
    color: theme.colors.text.primary,
    padding: 0,
  },
  recommendBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    marginHorizontal: theme.spacing.md,
    marginBottom: theme.spacing.sm,
    paddingHorizontal: theme.spacing.md,
    paddingVertical: theme.spacing.sm,
    backgroundColor: `${theme.colors.primary}10`,
    borderRadius: theme.layout.borderRadius.md,
    gap: theme.spacing.sm,
  },
  recommendText: {
    flex: 1,
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.secondary,
  },
  listContent: {
    paddingHorizontal: theme.spacing.md,
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
    backgroundColor: `${theme.colors.primary}15`,
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
  cardMetrics: {
    flexDirection: 'row',
    gap: theme.spacing.sm,
    marginTop: theme.spacing.sm,
  },
  metricTag: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 3,
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: theme.layout.borderRadius.full,
    backgroundColor: theme.colors.background.tertiary,
  },
  metricTagText: {
    fontSize: theme.typography.sizes.tiny,
    color: theme.colors.text.tertiary,
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
    backgroundColor: theme.colors.primary,
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

export default AlgorithmBrowseScreen;
