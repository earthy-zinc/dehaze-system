/**
 * 我的收藏 (L2)
 *
 * FavoriteAPI.getPage 列表 + 取消收藏
 */
import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  FlatList,
  StyleSheet,
  TouchableOpacity,
  RefreshControl,
  Alert,
} from 'react-native';
import { useFocusEffect, useNavigation } from '@react-navigation/native';
import { FavoriteAPI } from 'dehaze-sdk-js';
import type { FavoriteVO } from 'dehaze-sdk-js';
import Ionicons from 'react-native-vector-icons/Ionicons';

import { theme } from '@/theme';
import { AppHeader } from '@/layout';

const PAGE_SIZE = 20;

const TYPE_LABELS: Record<string, string> = {
  algorithm: '算法',
  result: '结果',
  dataset: '数据集',
  image: '图片',
  preset: '预设',
};

const PersonalFavoritesScreen: React.FC = () => {
  const navigation = useNavigation();
  const [favorites, setFavorites] = useState<FavoriteVO[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [page, setPage] = useState(1);

  const loadFavorites = useCallback(async (pageNum = 1, isRefresh = false) => {
    try {
      if (isRefresh) setRefreshing(true);
      else if (pageNum === 1) setLoading(true);
      const result = await FavoriteAPI.getPage({ pageNum, pageSize: PAGE_SIZE });
      const list = result.list || [];
      if (pageNum === 1) setFavorites(list);
      else setFavorites(prev => [...prev, ...list]);
      setHasMore(list.length >= PAGE_SIZE);
      setPage(pageNum);
    } catch {
      Alert.alert('加载失败', '获取收藏列表失败，请重试');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useFocusEffect(
    useCallback(() => {
      loadFavorites(1);
    }, [loadFavorites]),
  );

  const handleRefresh = useCallback(() => loadFavorites(1, true), [loadFavorites]);
  const handleLoadMore = useCallback(() => {
    if (hasMore && !refreshing) loadFavorites(page + 1);
  }, [hasMore, refreshing, page, loadFavorites]);

  const handleRemove = useCallback(
    (id: number) => {
      Alert.alert('取消收藏', '确认取消收藏此项？', [
        { text: '取消', style: 'cancel' },
        {
          text: '确认',
          style: 'destructive',
          onPress: async () => {
            try {
              await FavoriteAPI.deleteByIds([id]);
              setFavorites(prev => prev.filter(f => f.id !== id));
            } catch {
              Alert.alert('操作失败', '请稍后重试');
            }
          },
        },
      ]);
    },
    [],
  );

  const renderItem = useCallback(
    ({ item }: { item: FavoriteVO }) => {
      const typeLabel = TYPE_LABELS[item.targetType] || item.targetType;

      return (
        <View style={styles.card}>
          <View style={styles.cardBody}>
            <View style={styles.iconWrap}>
              <Ionicons
                name="heart"
                size={20}
                color={item.isInvalid ? theme.colors.text.tertiary : theme.colors.status.error}
              />
            </View>
            <View style={styles.info}>
              <Text style={[styles.name, item.isInvalid && styles.invalidText]} numberOfLines={1}>
                {item.targetName || '未命名'}
              </Text>
              <View style={styles.metaRow}>
                <View style={styles.typeBadge}>
                  <Text style={styles.typeText}>{typeLabel}</Text>
                </View>
                {item.isInvalid && (
                  <Text style={styles.invalidBadge}>已失效</Text>
                )}
                <Text style={styles.time}>
                  {item.createTime ? new Date(item.createTime).toLocaleDateString('zh-CN') : ''}
                </Text>
              </View>
            </View>
            <TouchableOpacity
              onPress={() => handleRemove(item.id)}
              hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
            >
              <Ionicons name="trash-outline" size={18} color={theme.colors.text.tertiary} />
            </TouchableOpacity>
          </View>
        </View>
      );
    },
    [handleRemove],
  );

  const renderEmpty = () =>
    !loading ? (
      <View style={styles.empty}>
        <Ionicons name="heart-outline" size={48} color={theme.colors.text.tertiary} />
        <Text style={styles.emptyText}>暂无收藏</Text>
      </View>
    ) : null;

  return (
    <View style={styles.container}>
      <AppHeader title="我的收藏" showBack onBackPress={() => navigation.goBack()} />
      <FlatList
        data={favorites}
        renderItem={renderItem}
        keyExtractor={item => String(item.id)}
        contentContainerStyle={styles.list}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={handleRefresh} colors={[theme.colors.primary]} tintColor={theme.colors.primary} />
        }
        onEndReached={handleLoadMore}
        onEndReachedThreshold={0.5}
        ListEmptyComponent={renderEmpty}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: theme.colors.background.secondary },
  list: { padding: theme.spacing.md, flexGrow: 1 },
  card: {
    backgroundColor: theme.colors.background.primary,
    borderRadius: theme.layout.borderRadius.md,
    padding: theme.spacing.md,
    marginBottom: theme.spacing.sm,
    ...theme.layout.shadows.sm,
  },
  cardBody: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.sm,
  },
  iconWrap: {
    width: 40,
    height: 40,
    borderRadius: 8,
    backgroundColor: theme.colors.background.tertiary,
    justifyContent: 'center',
    alignItems: 'center',
  },
  info: { flex: 1 },
  name: {
    fontSize: theme.typography.sizes.bodySmall,
    fontWeight: theme.typography.weights.medium,
    color: theme.colors.text.primary,
    marginBottom: 4,
  },
  invalidText: {
    color: theme.colors.text.tertiary,
  },
  metaRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  typeBadge: {
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
    backgroundColor: theme.colors.primaryLight,
  },
  typeText: {
    fontSize: theme.typography.sizes.tiny,
    color: theme.colors.primary,
    fontWeight: theme.typography.weights.medium,
  },
  invalidBadge: {
    fontSize: theme.typography.sizes.tiny,
    color: theme.colors.text.tertiary,
  },
  time: {
    fontSize: theme.typography.sizes.tiny,
    color: theme.colors.text.tertiary,
  },
  empty: { alignItems: 'center', paddingVertical: theme.spacing.xxxl },
  emptyText: { fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.tertiary, marginTop: theme.spacing.sm },
});

export default PersonalFavoritesScreen;
