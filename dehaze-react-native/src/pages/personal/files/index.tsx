/**
 * 我的文件 (L2)
 *
 * FileAPI.getPage 分页列表，展示个人上传文件
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
import { FileAPI } from 'dehaze-sdk-js';
import type { FileInfo } from 'dehaze-sdk-js';
import Ionicons from 'react-native-vector-icons/Ionicons';

import { theme } from '@/theme';
import { AppHeader } from '@/layout';
import { formatFileSize } from './utils';

const PAGE_SIZE = 20;

const PersonalFilesScreen: React.FC = () => {
  const navigation = useNavigation();
  const [files, setFiles] = useState<FileInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [page, setPage] = useState(1);

  const loadFiles = useCallback(async (pageNum = 1, isRefresh = false) => {
    try {
      if (isRefresh) setRefreshing(true);
      else if (pageNum === 1) setLoading(true);

      const result = await FileAPI.getPage({ pageNum, pageSize: PAGE_SIZE });
      const list = result.list || [];
      if (pageNum === 1) {
        setFiles(list);
      } else {
        setFiles(prev => [...prev, ...list]);
      }
      setHasMore(list.length >= PAGE_SIZE);
      setPage(pageNum);
    } catch {
      Alert.alert('加载失败', '获取文件列表失败，请重试');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useFocusEffect(
    useCallback(() => {
      loadFiles(1);
    }, [loadFiles]),
  );

  const handleRefresh = useCallback(() => loadFiles(1, true), [loadFiles]);
  const handleLoadMore = useCallback(() => {
    if (hasMore && !refreshing) loadFiles(page + 1);
  }, [hasMore, refreshing, page, loadFiles]);

  const handleDelete = useCallback(
    (fileId: number) => {
      Alert.alert('删除文件', '确认删除此文件？', [
        { text: '取消', style: 'cancel' },
        {
          text: '删除',
          style: 'destructive',
          onPress: async () => {
            try {
              await FileAPI.deleteById(fileId);
              setFiles(prev => prev.filter(f => f.id !== fileId));
            } catch {
              Alert.alert('删除失败', '请稍后重试');
            }
          },
        },
      ]);
    },
    [],
  );

  const renderItem = useCallback(
    ({ item }: { item: FileInfo }) => {
      const ext = (item.name || '').split('.').pop()?.toLowerCase() || '';
      const isImage = ['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp'].includes(ext);

      return (
        <View style={styles.fileRow}>
          <View style={styles.fileIcon}>
            <Ionicons
              name={isImage ? 'image-outline' : 'document-outline'}
              size={24}
              color={theme.colors.text.secondary}
            />
          </View>
          <View style={styles.fileInfo}>
            <Text style={styles.fileName} numberOfLines={1}>{item.name}</Text>
            <Text style={styles.fileMeta}>
              {item.size || formatFileSize(item.sizeBytes)}
              {item.createTime ? ` · ${new Date(item.createTime).toLocaleDateString('zh-CN')}` : ''}
            </Text>
          </View>
          <TouchableOpacity
            onPress={() => handleDelete(item.id)}
            hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
          >
            <Ionicons name="trash-outline" size={18} color={theme.colors.text.tertiary} />
          </TouchableOpacity>
        </View>
      );
    },
    [handleDelete],
  );

  const renderEmpty = () =>
    !loading ? (
      <View style={styles.empty}>
        <Ionicons name="document-outline" size={48} color={theme.colors.text.tertiary} />
        <Text style={styles.emptyText}>暂无文件</Text>
      </View>
    ) : null;

  return (
    <View style={styles.container}>
      <AppHeader title="我的文件" showBack onBackPress={() => navigation.goBack()} />
      <FlatList
        data={files}
        renderItem={renderItem}
        keyExtractor={item => String(item.id)}
        contentContainerStyle={styles.list}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={handleRefresh} colors={[theme.colors.primary]} tintColor={theme.colors.primary} />
        }
        onEndReached={handleLoadMore}
        onEndReachedThreshold={0.5}
        ListEmptyComponent={renderEmpty}
        ItemSeparatorComponent={ItemSeparator}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: theme.colors.background.secondary },
  list: { padding: theme.spacing.md, flexGrow: 1 },
  fileRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.sm,
    paddingVertical: theme.spacing.sm,
  },
  fileIcon: {
    width: 44,
    height: 44,
    borderRadius: theme.layout.borderRadius.sm,
    backgroundColor: theme.colors.background.tertiary,
    justifyContent: 'center',
    alignItems: 'center',
  },
  fileInfo: { flex: 1 },
  fileName: {
    fontSize: theme.typography.sizes.bodySmall,
    fontWeight: theme.typography.weights.medium,
    color: theme.colors.text.primary,
  },
  fileMeta: {
    fontSize: theme.typography.sizes.tiny,
    color: theme.colors.text.tertiary,
    marginTop: 2,
  },
  sep: { height: StyleSheet.hairlineWidth, backgroundColor: theme.colors.border.light },
  empty: { alignItems: 'center', paddingVertical: theme.spacing.xxxl },
  emptyText: { fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.tertiary, marginTop: theme.spacing.sm },
});

const ItemSeparator = () => <View style={styles.sep} />;

export default PersonalFilesScreen;
