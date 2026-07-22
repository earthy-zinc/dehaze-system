/**
 * 算法选择页面
 *
 * 接收来自图像输入页面的 SelectedImage，提供：
 * - 智能推荐（基于图像特征，由 Python 后端推荐 Top3）
 * - 树形算法浏览（由 Java 后端 AlgorithmAPI.getList 提供树）
 * - 收藏管理（Python 后端 algorithm-select/favorite）
 * - 算法对比（Python 后端 algorithm-select/compare，最多 3 个）
 *
 * 选中算法后导航到 Processing 页面（携带 algorithmId + image）。
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  ActivityIndicator,
  TouchableOpacity,
  Alert,
} from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '@/routes/types';
import { MainLayout } from '@/layout';
import { theme } from '@/theme';
import Icon from '@/components/Icon';
import type { SelectedImage } from '@/types/image';
import type { Algorithm, AlgorithmRecommendVO, FavoriteVO } from '@/types/algorithm';
import { AlgorithmAPI } from 'dehaze-sdk-js';
import AlgorithmSelectAPI from '@/api/algorithm-select';

import AlgorithmTree from './components/AlgorithmTree';
import AlgorithmCard from './components/AlgorithmCard';
import CompareBar from './components/CompareBar';
import CompareModal from './components/CompareModal';

type Props = NativeStackScreenProps<RootStackParamList, 'AlgorithmSelect'>;

type TabKey = 'recommend' | 'browse' | 'favorites';

const MAX_COMPARE = 3;

const AlgorithmSelectScreen: React.FC<Props> = ({ route, navigation }) => {
  const image: SelectedImage | undefined = route.params?.image;

  // 数据状态
  const [tree, setTree] = useState<Algorithm[]>([]);
  const [treeLoading, setTreeLoading] = useState(true);
  const [recommendations, setRecommendations] = useState<AlgorithmRecommendVO[]>([]);
  const [recommendLoading, setRecommendLoading] = useState(false);
  const [favorites, setFavorites] = useState<FavoriteVO[]>([]);
  const [favoriteIds, setFavoriteIds] = useState<Set<number>>(new Set());

  // UI 状态
  const [activeTab, setActiveTab] = useState<TabKey>(image ? 'recommend' : 'browse');
  const [compareIds, setCompareIds] = useState<Set<number>>(new Set());
  const [compareAlgorithms, setCompareAlgorithms] = useState<Algorithm[]>([]);
  const [showCompareModal, setShowCompareModal] = useState(false);

  /** 判断图片是否为远程 URL（后端可访问） */
  const hasRemoteImage = !!(image?.url && image.url.startsWith('http'));

  /** 收集树中所有叶子算法，用于查找 */
  const allLeafAlgorithms = useMemo(() => {
    const collect = (nodes: Algorithm[]): Algorithm[] => {
      const result: Algorithm[] = [];
      for (const node of nodes) {
        if (!node.children || node.children.length === 0) {
          result.push(node);
        } else {
          result.push(...collect(node.children));
        }
      }
      return result;
    };
    return collect(tree);
  }, [tree]);

  /** 根据 ID 查找算法 */
  const findAlgorithmById = useCallback(
    (id: number): Algorithm | undefined => {
      return allLeafAlgorithms.find(a => a.id === id);
    },
    [allLeafAlgorithms],
  );

  /**
   * 将 Python 后端返回的扁平 VO 解析为 Algorithm：
   * 优先从算法树中取完整信息，取不到时用 VO 字段构造最小对象
   */
  const resolveAlgorithm = useCallback(
    (id: number, name?: string, type?: string): Algorithm => {
      const found = findAlgorithmById(id);
      if (found) {
        return found;
      }
      return { id, parentId: 0, name: name ?? `算法${id}`, type: type ?? '', description: '' };
    },
    [findAlgorithmById],
  );

  // 加载算法树
  useEffect(() => {
    setTreeLoading(true);
    AlgorithmAPI.getList()
      .then(data => {
        setTree(data || []);
      })
      .catch(err => {
        Alert.alert('加载失败', err instanceof Error ? err.message : '无法加载算法列表');
      })
      .finally(() => setTreeLoading(false));
  }, []);

  // 加载推荐
  useEffect(() => {
    if (!hasRemoteImage || !image?.url) return;

    setRecommendLoading(true);
    AlgorithmSelectAPI.recommend({ imageUrl: image.url })
      .then(data => {
        setRecommendations(data || []);
      })
      .catch(() => {
        // 推荐服务不可用（Python 后端未启动）时静默失败，不阻塞页面
      })
      .finally(() => setRecommendLoading(false));
  }, [hasRemoteImage, image?.url]);

  // 加载收藏列表
  const loadFavorites = useCallback(() => {
    AlgorithmSelectAPI.listFavorites()
      .then(data => {
        const favList = data || [];
        setFavorites(favList);
        setFavoriteIds(new Set(favList.map(r => r.algorithmId)));
      })
      .catch(() => {
        // 收藏服务不可用（Python 后端未启动）时静默失败，不影响页面使用
      });
  }, []);

  useEffect(() => {
    loadFavorites();
  }, [loadFavorites]);

  /** 选择算法 → 导航到处理页面 */
  const handleSelect = useCallback(
    (algorithm: Algorithm) => {
      navigation.navigate('Processing', {
        algorithmId: algorithm.id,
        image,
      });
    },
    [navigation, image],
  );

  /** 查看算法详情 */
  const handleViewDetail = useCallback(
    (algorithm: Algorithm) => {
      navigation.navigate('Algorithm', { algorithmId: algorithm.id });
    },
    [navigation],
  );

  /** 切换收藏 */
  const handleToggleFavorite = useCallback(
    (algorithm: Algorithm) => {
      AlgorithmSelectAPI.toggleFavorite(algorithm.id)
        .then(res => {
          const isFav = res.favorited;
          setFavoriteIds(prev => {
            const next = new Set(prev);
            if (isFav) {
              next.add(algorithm.id);
            } else {
              next.delete(algorithm.id);
            }
            return next;
          });
          setFavorites(prev =>
            isFav
              ? prev.some(f => f.algorithmId === algorithm.id)
                ? prev
                : [
                    ...prev,
                    {
                      id: res.favoriteId ?? 0,
                      userId: 0,
                      algorithmId: algorithm.id,
                      algorithmName: algorithm.name,
                    },
                  ]
              : prev.filter(f => f.algorithmId !== algorithm.id),
          );
        })
        .catch(err => {
          Alert.alert('操作失败', err instanceof Error ? err.message : '收藏操作失败');
        });
    },
    [],
  );

  /** 切换对比选择 */
  const handleToggleCompare = useCallback(
    (algorithm: Algorithm) => {
      setCompareIds(prev => {
        const next = new Set(prev);
        if (next.has(algorithm.id)) {
          next.delete(algorithm.id);
        } else {
          if (next.size >= MAX_COMPARE) {
            Alert.alert('提示', `最多只能选择 ${MAX_COMPARE} 个算法进行对比`);
            return prev;
          }
          next.add(algorithm.id);
        }
        return next;
      });
    },
    [],
  );

  /** 执行对比 */
  const handleCompare = useCallback(() => {
    const selected = allLeafAlgorithms.filter(a => compareIds.has(a.id));
    if (selected.length < 2) return;
    setCompareAlgorithms(selected);
    setShowCompareModal(true);
  }, [allLeafAlgorithms, compareIds]);

  /** 清空对比选择 */
  const handleClearCompare = useCallback(() => {
    setCompareIds(new Set());
  }, []);

  /** Tab 配置 */
  const tabs: { key: TabKey; label: string; visible: boolean }[] = [
    { key: 'recommend', label: '推荐', visible: hasRemoteImage },
    { key: 'browse', label: '浏览', visible: true },
    { key: 'favorites', label: '收藏', visible: true },
  ];
  const visibleTabs = tabs.filter(t => t.visible);

  /** 渲染图片横幅 */
  const renderImageBanner = () => {
    if (!image) return null;
    return (
      <View style={styles.imageBanner}>
        <Icon name="image" size={20} color={theme.colors.primary} />
        <Text style={styles.bannerText} numberOfLines={1}>
          {image.name || '已选择图片'}
        </Text>
        {hasRemoteImage ? (
          <View style={styles.remoteBadge}>
            <Text style={styles.remoteBadgeText}>可推荐</Text>
          </View>
        ) : (
          <Text style={styles.localHint}>本地图片</Text>
        )}
      </View>
    );
  };

  /** 渲染 Tab 选择器 */
  const renderTabs = () => (
    <View style={styles.tabContainer}>
      {visibleTabs.map(tab => {
        const isActive = activeTab === tab.key;
        return (
          <TouchableOpacity
            key={tab.key}
            style={[styles.tab, isActive && styles.tabActive]}
            onPress={() => setActiveTab(tab.key)}
            activeOpacity={0.7}
          >
            <Text style={[styles.tabText, isActive && styles.tabTextActive]}>
              {tab.label}
            </Text>
          </TouchableOpacity>
        );
      })}
    </View>
  );

  /** 渲染推荐区域 */
  const renderRecommend = () => {
    if (recommendLoading) {
      return (
        <View style={styles.centerContainer}>
          <ActivityIndicator size="large" color={theme.colors.primary} />
          <Text style={styles.loadingText}>正在分析图片并生成推荐...</Text>
        </View>
      );
    }

    if (recommendations.length === 0) {
      return (
        <View style={styles.centerContainer}>
          <Icon name="bulb-outline" size={48} color={theme.colors.text.tertiary} />
          <Text style={styles.emptyText}>暂无推荐算法</Text>
          <Text style={styles.emptySubtext}>请切换到「浏览」标签手动选择</Text>
        </View>
      );
    }

    return (
      <View>
        <Text style={styles.sectionTitle}>
          为您推荐 Top {recommendations.length} 算法
        </Text>
        {recommendations.map(r => {
          const algorithm = resolveAlgorithm(r.algorithmId, r.algorithmName, r.type);
          return (
            <AlgorithmCard
              key={r.algorithmId}
              algorithm={algorithm}
              matchScore={Math.round(r.score)}
              reason={r.reason}
              isFavorite={favoriteIds.has(r.algorithmId)}
              isSelected={compareIds.has(r.algorithmId)}
              onSelect={handleSelect}
              onToggleFavorite={handleToggleFavorite}
              onViewDetail={handleViewDetail}
              onToggleCompare={handleToggleCompare}
            />
          );
        })}
      </View>
    );
  };

  /** 渲染浏览区域 */
  const renderBrowse = () => {
    if (treeLoading) {
      return (
        <View style={styles.centerContainer}>
          <ActivityIndicator size="large" color={theme.colors.primary} />
          <Text style={styles.loadingText}>正在加载算法列表...</Text>
        </View>
      );
    }

    return (
      <AlgorithmTree
        tree={tree}
        favoriteIds={favoriteIds}
        compareIds={compareIds}
        onSelect={handleSelect}
        onToggleFavorite={handleToggleFavorite}
        onViewDetail={handleViewDetail}
        onToggleCompare={handleToggleCompare}
      />
    );
  };

  /** 渲染收藏区域 */
  const renderFavorites = () => {
    if (favorites.length === 0) {
      return (
        <View style={styles.centerContainer}>
          <Icon name="heart-outline" size={48} color={theme.colors.text.tertiary} />
          <Text style={styles.emptyText}>暂无收藏算法</Text>
          <Text style={styles.emptySubtext}>在算法卡片中点击「收藏」即可添加</Text>
        </View>
      );
    }

    return (
      <View>
        <Text style={styles.sectionTitle}>已收藏 {favorites.length} 个算法</Text>
        {favorites.map(fav => {
          const algo = resolveAlgorithm(fav.algorithmId, fav.algorithmName);
          return (
            <AlgorithmCard
              key={fav.algorithmId}
              algorithm={algo}
              isFavorite={true}
              isSelected={compareIds.has(fav.algorithmId)}
              onSelect={handleSelect}
              onToggleFavorite={handleToggleFavorite}
              onViewDetail={handleViewDetail}
              onToggleCompare={handleToggleCompare}
            />
          );
        })}
      </View>
    );
  };

  /** 渲染当前 Tab 内容 */
  const renderContent = () => {
    switch (activeTab) {
      case 'recommend':
        return renderRecommend();
      case 'browse':
        return renderBrowse();
      case 'favorites':
        return renderFavorites();
      default:
        return null;
    }
  };

  return (
    <MainLayout title="算法选择">
      <View style={styles.container}>
        <ScrollView
          style={styles.scrollView}
          contentContainerStyle={[
            styles.scrollContent,
            compareIds.size > 0 && { paddingBottom: 80 },
          ]}
          showsVerticalScrollIndicator={false}
        >
          {/* 图片横幅 */}
          {renderImageBanner()}

          {/* Tab 选择器 */}
          {renderTabs()}

          {/* 内容区域 */}
          {renderContent()}
        </ScrollView>

        {/* 对比栏 */}
        <CompareBar
          selectedCount={compareIds.size}
          maxCount={MAX_COMPARE}
          onCompare={handleCompare}
          onClear={handleClearCompare}
        />

        {/* 对比弹窗 */}
        <CompareModal
          visible={showCompareModal}
          algorithms={compareAlgorithms}
          imageUrl={hasRemoteImage ? image?.url : undefined}
          onClose={() => setShowCompareModal(false)}
          onSelect={handleSelect}
        />
      </View>
    </MainLayout>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: theme.colors.background.secondary,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: theme.spacing.lg,
  },
  imageBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: theme.colors.background.primary,
    borderRadius: theme.layout.borderRadius.md,
    paddingHorizontal: theme.spacing.md,
    paddingVertical: theme.spacing.sm,
    marginBottom: theme.spacing.md,
    gap: theme.spacing.sm,
    ...theme.layout.shadows.sm,
  },
  bannerText: {
    flex: 1,
    fontSize: theme.typography.sizes.body,
    color: theme.colors.text.primary,
    fontWeight: theme.typography.weights.medium,
  },
  remoteBadge: {
    backgroundColor: theme.colors.status.success,
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 10,
  },
  remoteBadgeText: {
    fontSize: theme.typography.sizes.small,
    color: '#fff',
    fontWeight: theme.typography.weights.medium,
  },
  localHint: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.tertiary,
  },
  tabContainer: {
    flexDirection: 'row',
    backgroundColor: theme.colors.background.tertiary,
    borderRadius: theme.layout.borderRadius.md,
    padding: 4,
    marginBottom: theme.spacing.md,
  },
  tab: {
    flex: 1,
    paddingVertical: theme.spacing.sm,
    borderRadius: theme.layout.borderRadius.sm,
    alignItems: 'center',
  },
  tabActive: {
    backgroundColor: theme.colors.background.primary,
    ...theme.layout.shadows.sm,
  },
  tabText: {
    fontSize: theme.typography.sizes.body,
    fontWeight: theme.typography.weights.medium,
    color: theme.colors.text.secondary,
  },
  tabTextActive: {
    color: theme.colors.primary,
    fontWeight: theme.typography.weights.semibold,
  },
  centerContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: theme.spacing.xxxl,
    gap: theme.spacing.sm,
  },
  loadingText: {
    fontSize: theme.typography.sizes.body,
    color: theme.colors.text.secondary,
  },
  emptyText: {
    fontSize: theme.typography.sizes.body,
    fontWeight: theme.typography.weights.medium,
    color: theme.colors.text.secondary,
  },
  emptySubtext: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.tertiary,
  },
  sectionTitle: {
    fontSize: theme.typography.sizes.body,
    fontWeight: theme.typography.weights.semibold,
    color: theme.colors.text.primary,
    marginBottom: theme.spacing.sm,
  },
});

export default AlgorithmSelectScreen;
