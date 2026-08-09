import React, {
  useState,
  useEffect,
  useCallback,
  useMemo,
  useRef,
} from "react";
import { View, Text, Image, ScrollView, Input } from "@tarojs/components";
import Taro from "@tarojs/taro";
import { ArrowLeft, Search, Star, StarOutlined, Close } from "@taroify/icons";
import { AlgorithmAPI, RecommendationAPI, FavoriteAPI } from "dehaze-sdk-js";
import type {
  Algorithm,
  AlgorithmDetailVO,
  RecommendedAlgorithm,
  AlgorithmCompareVO,
} from "dehaze-sdk-js";
import EmptyState from "@/components/common/EmptyState";
import { useProcessStore } from "@/stores/process";
import { getErrorMessage } from "@/utils/error";
import AlgorithmDetailPopup from "./components/AlgorithmDetailPopup";
import {
  getTypeWeight,
  collectLeafAlgorithms,
  getSearchHistory,
  saveSearchHistory,
  clearSearchHistory,
  COMPARE_MAX,
} from "./utils";
import type { TreeNode } from "./utils";
import "./index.less";

const AlgorithmSelectPage: React.FC = () => {
  // ==================== 核心数据 ====================
  const [tree, setTree] = useState<TreeNode[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedKeys, setExpandedKeys] = useState<Set<number>>(new Set());

  // ==================== 推荐 ====================
  const [currentImageUrl, setCurrentImageUrl] = useState("");
  const [recommendations, setRecommendations] = useState<
    RecommendedAlgorithm[]
  >([]);
  const [recommendLoading, setRecommendLoading] = useState(false);
  const [imageAnalysis, setImageAnalysis] = useState<{
    hazeLevel: string;
    sceneType: string;
  } | null>(null);

  // ==================== 收藏 ====================
  const [favoriteIds, setFavoriteIds] = useState<Set<number>>(new Set());
  const [favoriteMap, setFavoriteMap] = useState<Map<number, number>>(
    new Map()
  );
  const [togglingIds, setTogglingIds] = useState<Set<number>>(new Set());

  // ==================== 搜索 ====================
  const [searchKeyword, setSearchKeyword] = useState("");
  const [searchResults, setSearchResults] = useState<TreeNode[] | null>(null);
  const [, setSearchLoading] = useState(false);
  const [searchHistory, setSearchHistory] = useState<string[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const searchTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // ==================== 筛选（预留） ====================

  // ==================== 详情 ====================
  const [detailAlgorithm, setDetailAlgorithm] =
    useState<AlgorithmDetailVO | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  // ==================== 对比 ====================
  const [compareList, setCompareList] = useState<TreeNode[]>([]);
  const [compareResult, setCompareResult] = useState<
    AlgorithmCompareVO[] | null
  >(null);
  const [compareLoading, setCompareLoading] = useState(false);
  const [showCompare, setShowCompare] = useState(false);

  // ==================== 自定义测试 ====================
  const [testResult, setTestResult] = useState<string | null>(null);
  const [testLoading, setTestLoading] = useState(false);

  // ==================== 加载算法树 ====================
  const fetchTree = useCallback(async () => {
    try {
      setLoading(true);
      const data = await AlgorithmAPI.tree();
      setTree(data || []);
      const firstLevelIds = (data || []).map((item) => item.id);
      setExpandedKeys(new Set(firstLevelIds));
    } catch (error: unknown) {
      Taro.showToast({
        title: getErrorMessage(error, "加载算法失败"),
        icon: "none",
      });
    } finally {
      setLoading(false);
    }
  }, []);

  // ==================== 加载收藏状态 ====================
  const fetchFavorites = useCallback(async () => {
    try {
      const page = await FavoriteAPI.getPage({
        targetType: "algorithm",
        pageNum: 1,
        pageSize: 200,
      });
      const ids = new Set<number>();
      const map = new Map<number, number>();
      if (page?.list) {
        for (const fav of page.list) {
          ids.add(fav.targetId);
          map.set(fav.targetId, fav.id);
        }
      }
      setFavoriteIds(ids);
      setFavoriteMap(map);
    } catch {
      // 收藏加载失败不影响主流程
    }
  }, []);

  useEffect(() => {
    fetchTree();
    fetchFavorites();
    setSearchHistory(getSearchHistory());
    const image = useProcessStore.getState().image;
    if (image) {
      setCurrentImageUrl(image.url || "");
    }
  }, [fetchTree, fetchFavorites]);

  // ==================== 智能推荐 ====================
  const hasRemoteImage = !!(
    currentImageUrl && currentImageUrl.startsWith("http")
  );

  useEffect(() => {
    if (!hasRemoteImage || recommendations.length > 0) return;
    setRecommendLoading(true);
    RecommendationAPI.analyze({ imageUrl: currentImageUrl })
      .then(async (analysis) => {
        setImageAnalysis({
          hazeLevel: analysis.hazeLevel || "",
          sceneType: analysis.sceneType || "",
        });
        if (!analysis.imageMd5) {
          setRecommendations([]);
          return;
        }
        const recs = await RecommendationAPI.getAlgorithmRecommendations({
          imageMd5: analysis.imageMd5,
        });
        setRecommendations(recs || []);
      })
      .catch(() => setRecommendations([]))
      .finally(() => setRecommendLoading(false));
  }, [hasRemoteImage, currentImageUrl, recommendations.length]);

  // ==================== 搜索（防抖 300ms + 对接 API） ====================
  const handleSearchInput = useCallback((value: string) => {
    setSearchKeyword(value);
    if (searchTimer.current) clearTimeout(searchTimer.current);

    if (!value.trim()) {
      setSearchResults(null);
      setShowHistory(false);
      return;
    }

    setSearchLoading(true);
    searchTimer.current = setTimeout(async () => {
      try {
        const results = await AlgorithmAPI.search(value.trim());
        setSearchResults(results || []);
      } catch {
        setSearchResults([]);
      } finally {
        setSearchLoading(false);
      }
    }, 300);
  }, []);

  const handleSearchSubmit = useCallback(
    (keyword: string) => {
      if (!keyword.trim()) return;
      const history = saveSearchHistory(keyword.trim());
      setSearchHistory(history);
      setShowHistory(false);
      handleSearchInput(keyword);
    },
    [handleSearchInput]
  );

  // ==================== 收藏切换 ====================
  const toggleFavorite = useCallback(
    async (algorithmId: number) => {
      if (togglingIds.has(algorithmId)) return;
      setTogglingIds((prev) => new Set(prev).add(algorithmId));

      try {
        const existed = favoriteIds.has(algorithmId);
        if (existed) {
          const favId = favoriteMap.get(algorithmId);
          if (favId) await FavoriteAPI.deleteByIds([favId]);
          setFavoriteIds((prev) => {
            const next = new Set(prev);
            next.delete(algorithmId);
            return next;
          });
          setFavoriteMap((prev) => {
            const next = new Map(prev);
            next.delete(algorithmId);
            return next;
          });
          Taro.showToast({ title: "已取消收藏", icon: "none" });
        } else {
          const favId = await FavoriteAPI.add({
            targetType: "algorithm",
            targetId: algorithmId,
          });
          setFavoriteIds((prev) => new Set(prev).add(algorithmId));
          setFavoriteMap((prev) => new Map(prev).set(algorithmId, favId));
          Taro.showToast({ title: "已收藏", icon: "none" });
        }
      } catch (error: unknown) {
        Taro.showToast({
          title: getErrorMessage(error, "操作失败"),
          icon: "none",
        });
      } finally {
        setTogglingIds((prev) => {
          const next = new Set(prev);
          next.delete(algorithmId);
          return next;
        });
      }
    },
    [favoriteIds, favoriteMap, togglingIds]
  );

  // ==================== 展开/收起 ====================
  const toggleExpand = useCallback((id: number) => {
    setExpandedKeys((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }, []);

  // ==================== 选择算法 ====================
  const handleSelectAlgorithm = useCallback((node: TreeNode) => {
    useProcessStore.getState().setAlgorithm(node as unknown as Algorithm);
    Taro.navigateTo({ url: "/pages/processing/index" });
  }, []);

  // ==================== 查看详情 ====================
  const handleShowDetail = useCallback(async (node: TreeNode) => {
    setDetailLoading(true);
    try {
      const detail = await AlgorithmAPI.getSelectDetail(node.id);
      setDetailAlgorithm(detail);
    } catch (error: unknown) {
      Taro.showToast({
        title: getErrorMessage(error, "加载详情失败"),
        icon: "none",
      });
    } finally {
      setDetailLoading(false);
    }
  }, []);

  // ==================== 自定义测试 ====================
  const handleCustomTest = useCallback(
    async (algorithmId: number) => {
      if (!currentImageUrl) {
        Taro.showToast({ title: "请先选择图片", icon: "none" });
        return;
      }
      setTestLoading(true);
      try {
        const result = await AlgorithmAPI.test(algorithmId, {
          imageUrl: currentImageUrl,
        });
        setTestResult(result?.resultUrl || "");
      } catch (error: unknown) {
        Taro.showToast({
          title: getErrorMessage(error, "测试失败"),
          icon: "none",
        });
      } finally {
        setTestLoading(false);
      }
    },
    [currentImageUrl]
  );

  // ==================== 算法对比 ====================
  const allLeafAlgorithms = useMemo(() => collectLeafAlgorithms(tree), [tree]);

  const toggleCompare = useCallback((node: TreeNode) => {
    setCompareList((prev) => {
      const exists = prev.find((c) => c.id === node.id);
      if (exists) {
        return prev.filter((c) => c.id !== node.id);
      }
      if (prev.length >= COMPARE_MAX) {
        Taro.showToast({
          title: `最多对比 ${COMPARE_MAX} 个算法`,
          icon: "none",
        });
        return prev;
      }
      return [...prev, node];
    });
  }, []);

  const handleCompare = useCallback(async () => {
    if (compareList.length < 2) {
      Taro.showToast({ title: "至少选择 2 个算法", icon: "none" });
      return;
    }
    setCompareLoading(true);
    try {
      const result = await AlgorithmAPI.compare({
        algorithmIds: compareList.map((c) => c.id),
        imageUrl: currentImageUrl || undefined,
      });
      setCompareResult(result || []);
      setShowCompare(true);
    } catch (error: unknown) {
      Taro.showToast({
        title: getErrorMessage(error, "对比失败"),
        icon: "none",
      });
    } finally {
      setCompareLoading(false);
    }
  }, [compareList, currentImageUrl]);

  // ==================== 筛选逻辑 ====================
  const filteredTree = useMemo(() => {
    if (searchResults !== null) return searchResults;
    // 筛选条件由后端 search API 处理，此处直接透传
    return tree;
  }, [tree, searchResults]);

  // ==================== 渲染推荐卡片 ====================
  const renderRecommendCard = (rec: RecommendedAlgorithm, index: number) => {
    const node = allLeafAlgorithms.find((a) => a.id === rec.algorithmId);
    const isFav = favoriteIds.has(rec.algorithmId);
    const matchScore = Math.round(rec.matchScore || 0);
    return (
      <View
        key={rec.algorithmId}
        className="recommend-card"
        onClick={() => {
          if (node) handleSelectAlgorithm(node);
        }}
      >
        <View className="recommend-rank">#{index + 1}</View>
        <View className="recommend-info">
          <View className="recommend-header">
            <Text className="recommend-name">{rec.algorithmName}</Text>
            <View
              className={`recommend-tag type-${getTypeWeight(node?.type || "") >= 9 ? "dl" : getTypeWeight(node?.type || "") >= 7 ? "hybrid" : "traditional"}`}
            >
              <Text>{node?.type || "算法"}</Text>
            </View>
          </View>
          {rec.reason && <Text className="recommend-reason">{rec.reason}</Text>}
          {rec.effectDescription && (
            <Text className="recommend-effect">{rec.effectDescription}</Text>
          )}
          <View className="match-score-bar">
            <View
              className="match-score-fill"
              style={{ width: `${matchScore}%` }}
            />
          </View>
        </View>
        <View
          className="fav-btn"
          onClick={(e) => {
            e.stopPropagation();
            toggleFavorite(rec.algorithmId);
          }}
        >
          {isFav ? (
            <Star size="18" color="var(--color-warning)" />
          ) : (
            <StarOutlined size="18" color="var(--color-text-muted)" />
          )}
        </View>
      </View>
    );
  };

  // ==================== 渲染树节点 ====================
  const renderTreeNode = (node: TreeNode, level: number): React.ReactNode => {
    const hasChildren = node.children && node.children.length > 0;
    const isExpanded = expandedKeys.has(node.id);
    const isLeaf = !hasChildren && node.leaf;
    const isFav = favoriteIds.has(node.id);
    const inCompare = compareList.some((c) => c.id === node.id);

    return (
      <View key={node.id}>
        <View
          className={`tree-node level-${level} ${isLeaf ? "leaf" : "branch"} selectable`}
          onClick={() => {
            if (hasChildren) {
              toggleExpand(node.id);
            } else {
              handleSelectAlgorithm(node);
            }
          }}
        >
          <View className="node-indent" style={{ width: `${level * 32}rpx` }} />
          {hasChildren ? (
            <View className="expand-icon">
              <Text>{isExpanded ? "▼" : "▶"}</Text>
            </View>
          ) : (
            <View className="leaf-icon">
              <Text>⚡</Text>
            </View>
          )}
          <View className="node-content">
            <View className="node-header">
              <Text className="node-name">{node.name}</Text>
              {node.type && (
                <Text className="node-type-label">{node.type}</Text>
              )}
            </View>
          </View>
          {isLeaf && (
            <View className="node-actions">
              <View
                className="action-icon"
                onClick={(e) => {
                  e.stopPropagation();
                  toggleFavorite(node.id);
                }}
              >
                {isFav ? (
                  <Star size="16" color="var(--color-warning)" />
                ) : (
                  <StarOutlined size="16" color="var(--color-text-muted)" />
                )}
              </View>
              <View
                className="action-icon"
                onClick={(e) => {
                  e.stopPropagation();
                  handleShowDetail(node);
                }}
              >
                <Text
                  style={{
                    fontSize: "22rpx",
                    color: "var(--color-text-secondary)",
                  }}
                >
                  详情
                </Text>
              </View>
              <View
                className={`action-icon ${inCompare ? "in-compare" : ""}`}
                onClick={(e) => {
                  e.stopPropagation();
                  toggleCompare(node);
                }}
              >
                <Text
                  style={{
                    fontSize: "20rpx",
                    color: inCompare
                      ? "var(--color-info)"
                      : "var(--color-text-muted)",
                  }}
                >
                  对比
                </Text>
              </View>
            </View>
          )}
        </View>
        {hasChildren && isExpanded && (
          <View className="tree-children">
            {node.children!.map((child) => renderTreeNode(child, level + 1))}
          </View>
        )}
      </View>
    );
  };

  // ==================== 对比面板 ====================
  const renderComparePanel = () => {
    if (!showCompare) return null;
    return (
      <View className="compare-overlay">
        <View className="compare-panel">
          <View className="compare-header">
            <Text className="compare-title">算法对比</Text>
            <View
              className="compare-close"
              onClick={() => setShowCompare(false)}
            >
              <Close size="20" color="var(--color-text-secondary)" />
            </View>
          </View>
          <ScrollView className="compare-body" scrollY>
            {compareLoading ? (
              <View className="loading-state">
                <Text>对比中...</Text>
              </View>
            ) : compareResult && compareResult.length > 0 ? (
              <View className="compare-table">
                <View className="compare-row header-row">
                  <View className="compare-cell label-cell">
                    <Text>指标</Text>
                  </View>
                  {compareResult.map((c) => (
                    <View key={c.algorithmId} className="compare-cell">
                      <Text className="compare-alg-name">
                        {c.algorithmName}
                      </Text>
                    </View>
                  ))}
                </View>
                <View className="compare-row">
                  <View className="compare-cell label-cell">
                    <Text>处理耗时</Text>
                  </View>
                  {compareResult.map((c) => (
                    <View key={c.algorithmId} className="compare-cell">
                      <Text>{c.time ? `${c.time}ms` : "-"}</Text>
                    </View>
                  ))}
                </View>
                {compareResult.some((c) => c.resultUrl) && (
                  <View className="compare-row">
                    <View className="compare-cell label-cell">
                      <Text>效果预览</Text>
                    </View>
                    {compareResult.map((c) => (
                      <View key={c.algorithmId} className="compare-cell">
                        {c.resultUrl ? (
                          <Image
                            src={c.resultUrl}
                            className="compare-preview-img"
                            mode="aspectFill"
                          />
                        ) : (
                          <Text>-</Text>
                        )}
                      </View>
                    ))}
                  </View>
                )}
              </View>
            ) : (
              <EmptyState type="empty" title="暂无对比数据" />
            )}
          </ScrollView>
        </View>
      </View>
    );
  };

  // ==================== 主渲染 ====================
  return (
    <View className="algorithm-select-page">
      {/* L2 导航栏：返回 + 标题 */}
      <View className="navbar">
        <View className="nav-back" onClick={() => Taro.navigateBack()}>
          <ArrowLeft size="20" color="var(--color-text-primary)" />
        </View>
        <Text className="nav-title">选择算法</Text>
      </View>

      <ScrollView className="main-scroll" scrollY>
        {/* 当前图片预览 */}
        {currentImageUrl && (
          <View className="current-image-section">
            <Text className="section-label">当前图片</Text>
            <View className="current-image-wrapper">
              <Image
                src={currentImageUrl}
                className="current-image"
                mode="aspectFill"
                lazyLoad
              />
            </View>
          </View>
        )}

        {/* 智能推荐 */}
        {hasRemoteImage &&
          (recommendLoading || recommendations.length > 0 || imageAnalysis) && (
            <View className="recommend-section">
              <Text className="section-label">智能推荐</Text>
              {imageAnalysis && (
                <View className="analysis-tags">
                  <Text className="analysis-tag">
                    雾霾: {imageAnalysis.hazeLevel}
                  </Text>
                  <Text className="analysis-tag">
                    场景: {imageAnalysis.sceneType}
                  </Text>
                </View>
              )}
              {recommendLoading ? (
                <View className="loading-state">
                  <Text>分析中...</Text>
                </View>
              ) : recommendations.length > 0 ? (
                <View className="recommend-list">
                  {recommendations.map((rec, idx) =>
                    renderRecommendCard(rec, idx)
                  )}
                </View>
              ) : (
                <View className="loading-state">
                  <Text>暂无推荐</Text>
                </View>
              )}
            </View>
          )}

        {/* 搜索栏 */}
        <View className="search-section">
          <View className="search-input-wrapper">
            <Search size="18" color="var(--color-text-muted)" />
            <Input
              className="search-input"
              placeholder="搜索算法名称、类型或描述"
              value={searchKeyword}
              onFocus={() => {
                setShowHistory(true);
                setSearchHistory(getSearchHistory());
              }}
              onBlur={() => setTimeout(() => setShowHistory(false), 200)}
              onConfirm={(e) => handleSearchSubmit(e.detail.value)}
              onInput={(e) => handleSearchInput(e.detail.value)}
            />
            {searchKeyword && (
              <View
                className="clear-btn"
                onClick={() => {
                  setSearchKeyword("");
                  setSearchResults(null);
                }}
              >
                <Text>×</Text>
              </View>
            )}
          </View>

          {/* 搜索历史 */}
          {showHistory && !searchKeyword && searchHistory.length > 0 && (
            <View className="search-history-panel">
              <View className="history-header">
                <Text className="history-title">搜索历史</Text>
                <View
                  className="history-clear"
                  onClick={() => {
                    clearSearchHistory();
                    setSearchHistory([]);
                  }}
                >
                  <Text>清空</Text>
                </View>
              </View>
              <View className="history-tags">
                {searchHistory.map((kw) => (
                  <View
                    key={kw}
                    className="history-tag"
                    onClick={() => {
                      setSearchKeyword(kw);
                      handleSearchSubmit(kw);
                    }}
                  >
                    <Text>{kw}</Text>
                  </View>
                ))}
              </View>
            </View>
          )}
        </View>

        {/* 对比栏 */}
        {compareList.length > 0 && (
          <View className="compare-bar">
            <View className="compare-bar-tags">
              {compareList.map((c) => (
                <View key={c.id} className="compare-bar-tag">
                  <Text>{c.name}</Text>
                  <View
                    className="compare-bar-remove"
                    onClick={() => toggleCompare(c)}
                  >
                    <Close size="12" color="var(--color-text-muted)" />
                  </View>
                </View>
              ))}
            </View>
            <View className="compare-bar-btn" onClick={handleCompare}>
              <Text>
                对比 ({compareList.length}/{COMPARE_MAX})
              </Text>
            </View>
          </View>
        )}

        {/* 算法树 */}
        <View className="algorithm-tree-wrapper">
          {loading ? (
            <View className="loading-state">
              <Text>加载中...</Text>
            </View>
          ) : filteredTree.length === 0 ? (
            <EmptyState
              type="search"
              title="未找到算法"
              description={searchKeyword ? "请尝试其他关键词" : "暂无可用算法"}
            />
          ) : (
            <View className="algorithm-tree">
              {filteredTree.map((node) => renderTreeNode(node, 0))}
            </View>
          )}
        </View>
      </ScrollView>

      {/* 详情弹窗 */}
      <AlgorithmDetailPopup
        algorithm={detailAlgorithm}
        isFavorite={
          detailAlgorithm ? favoriteIds.has(detailAlgorithm.id) : false
        }
        loading={detailLoading}
        testResult={testResult}
        testLoading={testLoading}
        hasImage={!!currentImageUrl}
        onClose={() => {
          setDetailAlgorithm(null);
          setTestResult(null);
        }}
        onToggleFavorite={() => {
          if (detailAlgorithm) toggleFavorite(detailAlgorithm.id);
        }}
        onSelect={() => {
          if (detailAlgorithm) {
            useProcessStore
              .getState()
              .setAlgorithm(detailAlgorithm as Algorithm);
            Taro.navigateTo({ url: "/pages/processing/index" });
          }
        }}
        onCustomTest={() => {
          if (detailAlgorithm) handleCustomTest(detailAlgorithm.id);
        }}
      />

      {/* 对比面板 */}
      {renderComparePanel()}
    </View>
  );
};

export default AlgorithmSelectPage;
