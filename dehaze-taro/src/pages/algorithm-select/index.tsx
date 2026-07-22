import React, { useState, useEffect, useCallback, useMemo } from "react";
import { View, Text, Image, ScrollView, Input } from "@tarojs/components";
import Taro from "@tarojs/taro";
import { ArrowLeft, Search, Star, StarOutlined, Info } from "@taroify/icons";
import { Popup, Button } from "@taroify/core";
import { AlgorithmAPI } from "dehaze-sdk-js";
import type { Algorithm } from "dehaze-sdk-js";
import EmptyState from "@/components/common/EmptyState";
import "./index.less";

// 算法状态：3=已发布，可选
const PUBLISHED_STATUS = 3;
const FAVORITE_STORAGE_KEY = "favorite_algorithms";

// 算法类型推荐权重（数值越高越优先推荐）
const TYPE_WEIGHT: Record<string, number> = {
  // 深度学习类
  cnn: 10,
  gan: 9,
  transformer: 10,
  深度学习: 10,
  deeplab: 9,
  // 混合类
  混合: 7,
  hybrid: 7,
  // 传统类
  传统: 5,
  dcp: 5,
  retinex: 5,
  暗通道: 5,
};

// 获取算法类型权重
function getTypeWeight(type: string): number {
  const lower = (type || "").toLowerCase();
  for (const key of Object.keys(TYPE_WEIGHT)) {
    if (lower.includes(key.toLowerCase())) return TYPE_WEIGHT[key];
  }
  return 6;
}

// 递归收集所有叶子算法
function collectLeafAlgorithms(nodes: Algorithm[]): Algorithm[] {
  const result: Algorithm[] = [];
  const walk = (list: Algorithm[]) => {
    for (const node of list) {
      if (node.children && node.children.length > 0) {
        walk(node.children);
      } else {
        result.push(node);
      }
    }
  };
  walk(nodes);
  return result;
}

// 状态信息
function getStatusInfo(status?: number) {
  switch (status) {
    case 0:
      return { label: "草稿", className: "status-draft" };
    case 1:
      return { label: "测试中", className: "status-testing" };
    case 2:
      return { label: "待审核", className: "status-pending" };
    case 3:
      return { label: "已发布", className: "status-published" };
    case 4:
      return { label: "已停用", className: "status-disabled" };
    case 5:
      return { label: "已归档", className: "status-archived" };
    default:
      return { label: "未知", className: "status-unknown" };
  }
}

const AlgorithmSelectPage: React.FC = () => {
  const [algorithms, setAlgorithms] = useState<Algorithm[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchKeyword, setSearchKeyword] = useState("");
  const [expandedKeys, setExpandedKeys] = useState<Set<number>>(new Set());
  const [currentImageUrl, setCurrentImageUrl] = useState("");
  const [favoriteIds, setFavoriteIds] = useState<Set<number>>(new Set());
  const [detailAlgorithm, setDetailAlgorithm] = useState<Algorithm | null>(
    null
  );

  // 加载算法树
  const fetchAlgorithms = useCallback(async () => {
    try {
      setLoading(true);
      const data = await AlgorithmAPI.getList();
      setAlgorithms(data || []);
      // 默认展开第一层分类
      const firstLevelIds = (data || []).map((item) => item.id);
      setExpandedKeys(new Set(firstLevelIds));
    } catch (error: any) {
      Taro.showToast({ title: error?.message || "加载算法失败", icon: "none" });
    } finally {
      setLoading(false);
    }
  }, []);

  // 加载当前图片和收藏列表
  useEffect(() => {
    try {
      const stored = Taro.getStorageSync("current_image");
      if (stored) {
        const imageData = JSON.parse(stored);
        setCurrentImageUrl(imageData.url || "");
      }
    } catch {
      // 没有图片数据，忽略
    }

    try {
      const favStr = Taro.getStorageSync(FAVORITE_STORAGE_KEY);
      if (favStr) {
        const favArr: number[] = JSON.parse(favStr);
        setFavoriteIds(new Set(favArr));
      }
    } catch {
      // 收藏数据读取失败，忽略
    }
  }, []);

  useEffect(() => {
    fetchAlgorithms();
  }, [fetchAlgorithms]);

  // 所有叶子算法
  const allLeafAlgorithms = useMemo(
    () => collectLeafAlgorithms(algorithms),
    [algorithms]
  );

  // 智能推荐：从已发布叶子算法中按类型权重排序取前3
  const recommendedAlgorithms = useMemo(() => {
    const published = allLeafAlgorithms.filter(
      (a) => a.status === PUBLISHED_STATUS
    );
    return published
      .sort((a, b) => getTypeWeight(b.type) - getTypeWeight(a.type))
      .slice(0, 3);
  }, [allLeafAlgorithms]);

  // 收藏的算法列表
  const favoriteAlgorithms = useMemo(() => {
    return allLeafAlgorithms.filter((a) => favoriteIds.has(a.id));
  }, [allLeafAlgorithms, favoriteIds]);

  // 切换展开/收起
  const toggleExpand = useCallback((id: number) => {
    setExpandedKeys((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }, []);

  // 切换收藏
  const toggleFavorite = useCallback((algorithm: Algorithm) => {
    setFavoriteIds((prev) => {
      const next = new Set(prev);
      if (next.has(algorithm.id)) {
        next.delete(algorithm.id);
        Taro.showToast({ title: "已取消收藏", icon: "none" });
      } else {
        next.add(algorithm.id);
        Taro.showToast({ title: "已添加到收藏", icon: "success" });
      }
      // 持久化
      Taro.setStorageSync(FAVORITE_STORAGE_KEY, JSON.stringify([...next]));
      return next;
    });
  }, []);

  // 递归搜索过滤
  const filterTree = useCallback(
    (nodes: Algorithm[], keyword: string): Algorithm[] => {
      if (!keyword) return nodes;
      const lower = keyword.toLowerCase();
      const result: Algorithm[] = [];
      for (const node of nodes) {
        const nameMatch = node.name?.toLowerCase().includes(lower);
        const descMatch = node.description?.toLowerCase().includes(lower);
        if (node.children && node.children.length > 0) {
          const filteredChildren = filterTree(node.children, keyword);
          if (filteredChildren.length > 0 || nameMatch) {
            result.push({ ...node, children: filteredChildren });
          }
        } else if (nameMatch || descMatch) {
          result.push(node);
        }
      }
      return result;
    },
    []
  );

  const filteredAlgorithms = useMemo(() => {
    return filterTree(algorithms, searchKeyword);
  }, [algorithms, searchKeyword, filterTree]);

  // 选择算法
  const handleSelectAlgorithm = useCallback((algorithm: Algorithm) => {
    if (algorithm.status !== PUBLISHED_STATUS) {
      Taro.showToast({ title: "该算法未发布，暂不可用", icon: "none" });
      return;
    }
    Taro.setStorageSync("selected_algorithm", JSON.stringify(algorithm));
    Taro.navigateTo({ url: "/pages/processing/index" });
  }, []);

  // 渲染推荐卡片
  const renderRecommendCard = (algorithm: Algorithm, index: number) => {
    const isFav = favoriteIds.has(algorithm.id);
    return (
      <View
        key={algorithm.id}
        className="recommend-card"
        onClick={() => handleSelectAlgorithm(algorithm)}
      >
        <View className="recommend-rank">#{index + 1}</View>
        <View className="recommend-info">
          <View className="recommend-header">
            <Text className="recommend-name">{algorithm.name}</Text>
            <View
              className={`recommend-tag type-${getTypeWeight(algorithm.type) >= 9 ? "dl" : getTypeWeight(algorithm.type) >= 7 ? "hybrid" : "traditional"}`}
            >
              <Text>{algorithm.type || "算法"}</Text>
            </View>
          </View>
          {algorithm.description && (
            <Text className="recommend-desc">{algorithm.description}</Text>
          )}
          <View className="recommend-reason">
            <Text>推荐算法 · 适合当前图片</Text>
          </View>
        </View>
        <View
          className="fav-btn"
          onClick={(e) => {
            e.stopPropagation();
            toggleFavorite(algorithm);
          }}
        >
          {isFav ? (
            <Star size="18" color="#f59e0b" />
          ) : (
            <StarOutlined size="18" color="#9ca3af" />
          )}
        </View>
      </View>
    );
  };

  // 渲染收藏卡片
  const renderFavoriteCard = (algorithm: Algorithm) => {
    const statusInfo = getStatusInfo(algorithm.status);
    return (
      <View key={algorithm.id} className="favorite-card">
        <View
          className="fav-card-content"
          onClick={() => handleSelectAlgorithm(algorithm)}
        >
          <Text className="fav-name">{algorithm.name}</Text>
          <View className={`status-tag ${statusInfo.className}`}>
            <Text>{statusInfo.label}</Text>
          </View>
        </View>
        <View className="fav-btn" onClick={() => toggleFavorite(algorithm)}>
          <Star size="16" color="#f59e0b" />
        </View>
      </View>
    );
  };

  // 渲染树节点
  const renderNode = (node: Algorithm, level: number): React.ReactNode => {
    const hasChildren = node.children && node.children.length > 0;
    const isExpanded = expandedKeys.has(node.id);
    const isLeaf = !hasChildren;
    const isPublished = node.status === PUBLISHED_STATUS;
    const statusInfo = getStatusInfo(node.status);
    const isFav = favoriteIds.has(node.id);

    return (
      <View key={node.id}>
        <View
          className={`tree-node level-${level} ${isLeaf ? "leaf" : "branch"} ${isPublished ? "selectable" : ""}`}
          onClick={() => {
            if (hasChildren) {
              toggleExpand(node.id);
            } else {
              handleSelectAlgorithm(node);
            }
          }}
        >
          <View className="node-indent" style={{ width: `${level * 16}px` }} />
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
              {isLeaf && (
                <View className={`status-tag ${statusInfo.className}`}>
                  <Text>{statusInfo.label}</Text>
                </View>
              )}
            </View>
            {node.description && (
              <Text className="node-desc">{node.description}</Text>
            )}
            {isLeaf && (node.version || node.size || node.flops) && (
              <View className="node-meta">
                {node.version && (
                  <Text className="meta-text">v{node.version}</Text>
                )}
                {node.size && <Text className="meta-text">{node.size}</Text>}
                {node.flops && <Text className="meta-text">{node.flops}</Text>}
              </View>
            )}
            {isLeaf && node.type && (
              <View className="node-type">
                <Text className="type-label">{node.type}</Text>
              </View>
            )}
          </View>
          {/* 叶子节点操作按钮 */}
          {isLeaf && (
            <View className="node-actions">
              <View
                className="action-icon"
                onClick={(e) => {
                  e.stopPropagation();
                  toggleFavorite(node);
                }}
              >
                {isFav ? (
                  <Star size="16" color="#f59e0b" />
                ) : (
                  <StarOutlined size="16" color="#9ca3af" />
                )}
              </View>
              <View
                className="action-icon"
                onClick={(e) => {
                  e.stopPropagation();
                  setDetailAlgorithm(node);
                }}
              >
                <Info size="16" color="#6b7280" />
              </View>
              {isPublished && (
                <View className="select-btn">
                  <Text>使用</Text>
                </View>
              )}
            </View>
          )}
        </View>
        {hasChildren && isExpanded && (
          <View className="tree-children">
            {node.children!.map((child) => renderNode(child, level + 1))}
          </View>
        )}
      </View>
    );
  };

  return (
    <View className="algorithm-select-page">
      {/* 顶部导航 */}
      <View className="navbar">
        <View className="nav-back" onClick={() => Taro.navigateBack()}>
          <ArrowLeft size="20" color="#333" />
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

        {/* 智能推荐区域 */}
        {!searchKeyword && recommendedAlgorithms.length > 0 && (
          <View className="recommend-section">
            <View className="section-header">
              <Text className="section-title">智能推荐</Text>
              <Text className="section-hint">基于当前图片分析</Text>
            </View>
            <View className="recommend-list">
              {recommendedAlgorithms.map((algo, idx) =>
                renderRecommendCard(algo, idx)
              )}
            </View>
          </View>
        )}

        {/* 收藏区域 */}
        {!searchKeyword && favoriteAlgorithms.length > 0 && (
          <View className="favorite-section">
            <View className="section-header">
              <Text className="section-title">我的收藏</Text>
              <Text className="section-hint">
                {favoriteAlgorithms.length} 个算法
              </Text>
            </View>
            <View className="favorite-list">
              {favoriteAlgorithms.map((algo) => renderFavoriteCard(algo))}
            </View>
          </View>
        )}

        {/* 搜索栏 */}
        <View className="search-section">
          <View className="search-input-wrapper">
            <Search size="18" color="#9ca3af" />
            <Input
              className="search-input"
              placeholder="搜索算法名称或描述"
              value={searchKeyword}
              onInput={(e) => setSearchKeyword(e.detail.value)}
            />
            {searchKeyword && (
              <View className="clear-btn" onClick={() => setSearchKeyword("")}>
                <Text>×</Text>
              </View>
            )}
          </View>
        </View>

        {/* 算法树 */}
        <View className="algorithm-tree-wrapper">
          {loading ? (
            <View className="loading-state">
              <Text>加载中...</Text>
            </View>
          ) : filteredAlgorithms.length === 0 ? (
            <EmptyState
              type="search"
              title="未找到算法"
              description="请尝试其他关键词"
            />
          ) : (
            <View className="algorithm-tree">
              {filteredAlgorithms.map((node) => renderNode(node, 0))}
            </View>
          )}
        </View>
      </ScrollView>

      {/* 算法详情弹窗 */}
      <Popup
        open={!!detailAlgorithm}
        placement="bottom"
        style={{ height: "60%", borderRadius: "16px 16px 0 0" }}
        onClose={() => setDetailAlgorithm(null)}
      >
        {detailAlgorithm && (
          <View className="detail-popup">
            <View className="detail-header">
              <Text className="detail-title">算法详情</Text>
              <View
                className="detail-close"
                onClick={() => setDetailAlgorithm(null)}
              >
                <Text>✕</Text>
              </View>
            </View>
            <ScrollView className="detail-body" scrollY>
              <View className="detail-item">
                <Text className="detail-label">算法名称</Text>
                <Text className="detail-value">{detailAlgorithm.name}</Text>
              </View>
              {detailAlgorithm.type && (
                <View className="detail-item">
                  <Text className="detail-label">算法类型</Text>
                  <Text className="detail-value">{detailAlgorithm.type}</Text>
                </View>
              )}
              {detailAlgorithm.version && (
                <View className="detail-item">
                  <Text className="detail-label">版本</Text>
                  <Text className="detail-value">
                    {detailAlgorithm.version}
                  </Text>
                </View>
              )}
              {detailAlgorithm.size && (
                <View className="detail-item">
                  <Text className="detail-label">模型大小</Text>
                  <Text className="detail-value">{detailAlgorithm.size}</Text>
                </View>
              )}
              {detailAlgorithm.flops && (
                <View className="detail-item">
                  <Text className="detail-label">计算量</Text>
                  <Text className="detail-value">{detailAlgorithm.flops}</Text>
                </View>
              )}
              <View className="detail-item">
                <Text className="detail-label">状态</Text>
                <View
                  className={`status-tag ${getStatusInfo(detailAlgorithm.status).className}`}
                >
                  <Text>{getStatusInfo(detailAlgorithm.status).label}</Text>
                </View>
              </View>
              {detailAlgorithm.description && (
                <View className="detail-item detail-desc-item">
                  <Text className="detail-label">描述</Text>
                  <Text className="detail-value detail-desc">
                    {detailAlgorithm.description}
                  </Text>
                </View>
              )}
              {detailAlgorithm.createTime && (
                <View className="detail-item">
                  <Text className="detail-label">创建时间</Text>
                  <Text className="detail-value">
                    {detailAlgorithm.createTime}
                  </Text>
                </View>
              )}
            </ScrollView>
            <View className="detail-footer">
              <Button
                variant="outlined"
                onClick={() => toggleFavorite(detailAlgorithm)}
              >
                {favoriteIds.has(detailAlgorithm.id) ? "取消收藏" : "收藏"}
              </Button>
              {detailAlgorithm.status === PUBLISHED_STATUS && (
                <Button
                  color="primary"
                  onClick={() => {
                    setDetailAlgorithm(null);
                    handleSelectAlgorithm(detailAlgorithm);
                  }}
                >
                  立即使用
                </Button>
              )}
            </View>
          </View>
        )}
      </Popup>
    </View>
  );
};

export default AlgorithmSelectPage;
