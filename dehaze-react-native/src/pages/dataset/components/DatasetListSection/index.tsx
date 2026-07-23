import React, { useState, useCallback, useRef, useEffect } from 'react';
import {
  View,
  Text,
  FlatList,
  StyleSheet,
  RefreshControl,
  TouchableOpacity,
  ActivityIndicator,
} from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import SearchBar from '../SearchBar';
import LoadingSpinner from '@/components/LoadingSpinner';
import EmptyState from '@/components/EmptyState';
import Icon from '@/components/Icon';
import { DatasetTreeNode, Dataset } from '../../types/dataset';
import { datasetApi } from '../../services/datasetApi';

interface DatasetListSectionProps {
  onDatasetPress: (dataset: DatasetTreeNode) => void;
  searchValue: string;
  onSearchChange: (text: string) => void;
}

/** 根节点 parentId（后端约定 0 表示根） */
const ROOT_PARENT_ID = 0;

/** 递归更新节点 expanded */
const updateNodeExpanded = (
  nodes: DatasetTreeNode[],
  id: number,
  expanded: boolean,
): DatasetTreeNode[] =>
  nodes.map(n => {
    if (n.id === id) return { ...n, expanded };
    if (n.children)
      return { ...n, children: updateNodeExpanded(n.children, id, expanded) };
    return n;
  });

/** 递归更新节点 children */
const updateNodeChildren = (
  nodes: DatasetTreeNode[],
  id: number,
  children: DatasetTreeNode[],
  childrenLoaded: boolean,
): DatasetTreeNode[] =>
  nodes.map(n => {
    if (n.id === id) return { ...n, children, childrenLoaded };
    if (n.children)
      return {
        ...n,
        children: updateNodeChildren(n.children, id, children, childrenLoaded),
      };
    return n;
  });

const DatasetListSection: React.FC<DatasetListSectionProps> = ({
  onDatasetPress,
  searchValue,
  onSearchChange,
}) => {
  const [treeNodes, setTreeNodes] = useState<DatasetTreeNode[]>([]);
  const [isLoading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loadingKeys, setLoadingKeys] = useState<Set<number>>(new Set());
  /** 搜索防抖计时器 */
  const searchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  /** 将后端 Dataset[] 转为树节点（懒加载模式：丢弃后端 children，按需 fetchChildren） */
  const toTreeNodes = useCallback(
    (list: Dataset[], level: number): DatasetTreeNode[] => {
      return list.map(item => {
        const { children: _children, ...rest } = item;
        return {
          ...rest,
          level,
          expanded: false,
          childrenLoaded: false,
        };
      });
    },
    [],
  );

  /** 加载根节点（或搜索结果） */
  const loadRoot = useCallback(
    async (isRefresh = false) => {
      try {
        if (isRefresh) {
          setRefreshing(true);
        } else {
          setLoading(true);
        }
        setError(null);

        let rootList: Dataset[];
        if (searchValue.trim()) {
          // 搜索走分页接口，取前 50 条
          const page = await datasetApi.fetchDatasets({
            keyword: searchValue.trim(),
            pageNum: 1,
            pageSize: 50,
          });
          rootList = page.list || [];
        } else {
          // 非搜索走懒加载根节点
          rootList = await datasetApi.fetchChildren(ROOT_PARENT_ID);
        }

        const nodes = toTreeNodes(rootList, 0);
        setTreeNodes(nodes);
      } catch (err: unknown) {
        const e = err as { msg?: string; message?: string };
        const msg = e?.msg || e?.message || '加载数据集失败';
        setError(msg);
      } finally {
        setLoading(false);
        setRefreshing(false);
      }
    },
    [searchValue, toTreeNodes],
  );

  /** 搜索防抖 */
  useEffect(() => {
    if (searchTimerRef.current) {
      clearTimeout(searchTimerRef.current);
    }
    searchTimerRef.current = setTimeout(() => {
      loadRoot();
    }, 350);
    return () => {
      if (searchTimerRef.current) {
        clearTimeout(searchTimerRef.current);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchValue]);

  useFocusEffect(
    useCallback(() => {
      loadRoot();
    }, [loadRoot]),
  );

  /** 扁平化展开的节点（用于 FlatList 渲染） */
  const flattenTree = useCallback((): DatasetTreeNode[] => {
    const result: DatasetTreeNode[] = [];
    const walk = (nodes: DatasetTreeNode[]) => {
      for (const node of nodes) {
        result.push(node);
        if (node.expanded && node.children && node.children.length > 0) {
          walk(node.children);
        }
      }
    };
    walk(treeNodes);
    return result;
  }, [treeNodes]);

  /** 切换节点展开/收起 */
  const handleToggleExpand = useCallback(
    async (node: DatasetTreeNode) => {
      // 收起
      if (node.expanded) {
        setTreeNodes(prev => updateNodeExpanded(prev, node.id, false));
        return;
      }

      // 展开：若未加载子节点则先加载
      if (!node.childrenLoaded) {
        setLoadingKeys(prev => new Set(prev).add(node.id));
        try {
          const children = await datasetApi.fetchChildren(node.id);
          const childNodes = toTreeNodes(children || [], node.level + 1);
          setTreeNodes(prev =>
            updateNodeChildren(prev, node.id, childNodes, true),
          );
        } catch (err: unknown) {
          const e = err as { msg?: string; message?: string };
          setError(e?.msg || e?.message || '加载子数据集失败');
        } finally {
          setLoadingKeys(prev => {
            const next = new Set(prev);
            next.delete(node.id);
            return next;
          });
        }
      }

      setTreeNodes(prev => updateNodeExpanded(prev, node.id, true));
    },
    [toTreeNodes],
  );

  const handleRefresh = useCallback(() => {
    loadRoot(true);
  }, [loadRoot]);

  const flatNodes = flattenTree();

  const renderItem = useCallback(
    ({ item }: { item: DatasetTreeNode }) => {
      const hasChildren = item.hasChildren !== false;
      const isExpanded = !!item.expanded;
      const isLoadingChildren = loadingKeys.has(item.id);
      const indent = item.level * 20;

      return (
        <View style={[styles.nodeRow, { paddingLeft: 12 + indent }]}>
          {/* 展开/收起按钮 */}
          {hasChildren ? (
            <TouchableOpacity
              style={styles.expandBtn}
              onPress={() => handleToggleExpand(item)}
              hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
            >
              {isLoadingChildren ? (
                <ActivityIndicator size="small" color="#14b8a6" />
              ) : (
                <Icon
                  name={isExpanded ? 'chevron-down' : 'chevron-right'}
                  size={16}
                  color="#6b7280"
                />
              )}
            </TouchableOpacity>
          ) : (
            <View style={styles.leafPlaceholder} />
          )}

          {/* 节点内容 */}
          <TouchableOpacity
            style={styles.nodeContent}
            onPress={() => onDatasetPress(item)}
            activeOpacity={0.7}
          >
            <Icon name="database" size={18} color="#14b8a6" />
            <View style={styles.nodeTextWrap}>
              <Text style={styles.nodeName} numberOfLines={1}>
                {item.name}
              </Text>
              {!!item.type && (
                <Text style={styles.nodeType} numberOfLines={1}>
                  {item.type}
                </Text>
              )}
            </View>
            <Icon name="chevron-right" size={14} color="#d1d5db" />
          </TouchableOpacity>
        </View>
      );
    },
    [handleToggleExpand, onDatasetPress, loadingKeys],
  );

  const keyExtractor = useCallback((item: DatasetTreeNode) => item.id.toString(), []);

  if (error && treeNodes.length === 0 && !isLoading) {
    return (
      <View style={styles.container}>
        <View style={styles.searchContainer}>
          <SearchBar value={searchValue} onChangeText={onSearchChange} />
        </View>
        <EmptyState icon="search-plus" title="加载失败" description={error} />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.searchContainer}>
        <SearchBar value={searchValue} onChangeText={onSearchChange} />
      </View>

      <FlatList
        data={flatNodes}
        renderItem={renderItem}
        keyExtractor={keyExtractor}
        contentContainerStyle={styles.listContainer}
        showsVerticalScrollIndicator={false}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={handleRefresh}
            tintColor="#14b8a6"
            colors={['#14b8a6']}
          />
        }
        ListEmptyComponent={
          isLoading ? null : (
            <EmptyState
              icon="database"
              title="暂无数据集"
              description={
                searchValue ? '未找到匹配的数据集' : '还没有添加任何数据集'
              }
            />
          )
        }
        ListFooterComponent={
          isLoading ? (
            <View style={styles.loadingContainer}>
              <LoadingSpinner size="large" color="#14b8a6" />
            </View>
          ) : null
        }
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f9fafb',
  },
  searchContainer: {
    backgroundColor: '#ffffff',
    paddingHorizontal: 20,
    paddingTop: 16,
    paddingBottom: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#f3f4f6',
  },
  listContainer: {
    paddingVertical: 8,
  },
  nodeRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    paddingRight: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#f3f4f6',
    backgroundColor: '#ffffff',
  },
  expandBtn: {
    width: 28,
    height: 28,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 4,
  },
  leafPlaceholder: {
    width: 28,
    marginRight: 4,
  },
  nodeContent: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  nodeTextWrap: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  nodeName: {
    fontSize: 15,
    fontWeight: '600',
    color: '#1f2937',
  },
  nodeType: {
    fontSize: 12,
    color: '#9ca3af',
    backgroundColor: '#f3f4f6',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
    overflow: 'hidden',
  },
  loadingContainer: {
    paddingVertical: 20,
  },
});

export default DatasetListSection;
