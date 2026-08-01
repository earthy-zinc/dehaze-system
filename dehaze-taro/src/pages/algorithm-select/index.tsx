import React, { useState, useEffect, useCallback, useMemo } from "react";
import { View, Text, Image, ScrollView, Input } from "@tarojs/components";
import Taro from "@tarojs/taro";
import TaroCore from "@taroify/core";
import { Search, Star, StarOutlined } from "@taroify/icons";
import CompareNavbar from "@/components/compare/CompareNavbar";
import { AlgorithmAPI, RecommendationAPI } from "dehaze-sdk-js";
import type { Algorithm } from "dehaze-sdk-js";
import EmptyState from "@/components/common/EmptyState";
import { getErrorMessage } from "@/utils/error";
import AlgorithmDetailPopup from "./components/AlgorithmDetailPopup";
import AlgorithmTreeNode from "./components/AlgorithmTreeNode";
import {
  PUBLISHED_STATUS,
  FAVORITE_STORAGE_KEY,
  getTypeWeight,
  collectLeafAlgorithms,
  getStatusInfo,
  filterTree,
} from "./utils";
import "./index.less";

const Tabs = (TaroCore as any).Tabs;
const TabPanel = (TaroCore as any).TabPanel;

interface AlgorithmRecommendVO {
  algorithmId: number;
  algorithmName: string;
  score: number;
  reason?: string;
  type?: string;
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
  const [recommendations, setRecommendations] = useState<
    AlgorithmRecommendVO[]
  >([]);
  const [recommendLoading, setRecommendLoading] = useState(false);

  // 加载算法树
  const fetchAlgorithms = useCallback(async () => {
    try {
      setLoading(true);
      const data = await AlgorithmAPI.getList();
      setAlgorithms(data || []);
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

  // 智能推荐：使用RecommendationAPI获取推荐算法
  const hasRemoteImage = !!(
    currentImageUrl && currentImageUrl.startsWith("http")
  );

  useEffect(() => {
    if (!hasRemoteImage || recommendations.length > 0) return;
    setRecommendLoading(true);
    RecommendationAPI.analyze({ imageUrl: currentImageUrl })
      .then(async (analysis) => {
        const imageMd5 = analysis.imageMd5;
        if (!imageMd5) {
          setRecommendations([]);
          return;
        }
        const recs = await RecommendationAPI.getAlgorithmRecommendations({
          imageMd5,
        });
        setRecommendations(
          (recs || []).map((r) => ({
            algorithmId: r.algorithmId,
            algorithmName: r.algorithmName,
            score: r.matchScore,
            reason: r.reason,
            type: "",
          }))
        );
      })
      .catch(() => setRecommendations([]))
      .finally(() => setRecommendLoading(false));
  }, [hasRemoteImage, currentImageUrl]);

  // 根据推荐 VO 解析出完整 Algorithm
  const resolveRecommendAlgorithm = useCallback(
    (vo: AlgorithmRecommendVO): Algorithm => {
      const found = allLeafAlgorithms.find((a) => a.id === vo.algorithmId);
      if (found) return found;
      return {
        id: vo.algorithmId,
        parentId: 0,
        name: vo.algorithmName || `算法${vo.algorithmId}`,
        type: vo.type || "",
        description: "",
      } as Algorithm;
    },
    [allLeafAlgorithms]
  );

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
      Taro.setStorageSync(FAVORITE_STORAGE_KEY, JSON.stringify([...next]));
      return next;
    });
  }, []);

  const filteredAlgorithms = useMemo(() => {
    return filterTree(algorithms, searchKeyword);
  }, [algorithms, searchKeyword]);

  // 选择算法
  const handleSelectAlgorithm = useCallback((algorithm: Algorithm) => {
    if (algorithm.status !== PUBLISHED_STATUS) {
      Taro.showToast({ title: "该算法未发布，暂不可用", icon: "none" });
      return;
    }
    Taro.setStorageSync("selected_algorithm", JSON.stringify(algorithm));
    Taro.navigateTo({ url: "/pages/processing/index" });
  }, []);

  // 渲染推荐卡片（带匹配度进度条）
  const renderRecommendCard = (vo: AlgorithmRecommendVO, index: number) => {
    const algorithm = resolveRecommendAlgorithm(vo);
    const isFav = favoriteIds.has(algorithm.id);
    const matchScore = Math.round(vo.score || 0);
    const reasonText = vo.reason
      ? `${vo.reason} · 匹配度 ${matchScore}%`
      : `匹配度 ${matchScore}%`;
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
            <Text>{reasonText}</Text>
          </View>
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

  // 智能推荐Tab内容
  const renderRecommendTab = () => (
    <ScrollView className="recommend-tab-content" scrollY>
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

      {/* 推荐列表 */}
      <View className="recommend-list">
        {recommendLoading ? (
          <View className="loading-state">
            <Text>正在分析图片并生成推荐...</Text>
          </View>
        ) : recommendations.length > 0 ? (
          recommendations.map((vo, idx) => renderRecommendCard(vo, idx))
        ) : (
          <View className="loading-state">
            <Text>暂无推荐算法，请从下方算法树选择</Text>
          </View>
        )}
      </View>
    </ScrollView>
  );

  // 算法列表Tab内容
  const renderAlgorithmTab = () => (
    <ScrollView className="algorithm-tab-content" scrollY>
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
            {filteredAlgorithms.map((node) => (
              <AlgorithmTreeNode
                key={node.id}
                node={node}
                level={0}
                expandedKeys={expandedKeys}
                favoriteIds={favoriteIds}
                onToggleExpand={toggleExpand}
                onSelect={handleSelectAlgorithm}
                onToggleFavorite={toggleFavorite}
                onShowDetail={setDetailAlgorithm}
              />
            ))}
          </View>
        )}
      </View>
    </ScrollView>
  );

  return (
    <View className="algorithm-select-page">
      {/* 顶部导航 */}
      <CompareNavbar title="选择算法" />

      {/* 使用Tabs组件切换智能推荐和算法列表 */}
      <Tabs defaultValue="recommend">
        <TabPanel value="recommend">{renderRecommendTab()}</TabPanel>
        <TabPanel value="list">{renderAlgorithmTab()}</TabPanel>
      </Tabs>

      {/* 算法详情弹窗 */}
      <AlgorithmDetailPopup
        algorithm={detailAlgorithm}
        isFavorite={
          detailAlgorithm ? favoriteIds.has(detailAlgorithm.id) : false
        }
        onClose={() => setDetailAlgorithm(null)}
        onToggleFavorite={toggleFavorite}
        onSelect={handleSelectAlgorithm}
      />
    </View>
  );
};

export default AlgorithmSelectPage;
