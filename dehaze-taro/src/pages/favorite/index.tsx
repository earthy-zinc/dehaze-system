import React, { useState, useEffect, useCallback } from "react";
import { View, Text, ScrollView, Image } from "@tarojs/components";
import Taro from "@tarojs/taro";

import { Star } from "@taroify/icons";
import { FavoriteAPI, FavoriteTargetType } from "dehaze-sdk-js";
import type { FavoriteVO } from "dehaze-sdk-js";
import SearchBar from "@/components/common/SearchBar";
import EmptyState from "@/components/common/EmptyState";
import FavoriteButton from "@/components/favorite/FavoriteButton";
import PageLayout from "@/layout";
import StatusTag from "@/components/common/StatusTag";
import "./index.less";

// 类型标签配置
const TYPE_CONFIG: Record<
  FavoriteTargetType,
  { label: string; color: string }
> = {
  algorithm: { label: "算法", color: "#3b82f6" },
  result: { label: "结果", color: "#10b981" },
  dataset: { label: "数据集", color: "#f59e0b" },
  image: { label: "图片", color: "#ec4899" },
  preset: { label: "预设", color: "#7c3aed" },
};

const ALL_TYPES: { key: "" | FavoriteTargetType; label: string }[] = [
  { key: "", label: "全部" },
  { key: "algorithm", label: "算法" },
  { key: "result", label: "结果" },
  { key: "dataset", label: "数据集" },
  { key: "image", label: "图片" },
  { key: "preset", label: "预设" },
];

interface FavoriteItem extends FavoriteVO {
  targetType: FavoriteTargetType;
}

const FavoritePage: React.FC = () => {
  const [favorites, setFavorites] = useState<FavoriteItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [typeFilter, setTypeFilter] = useState<"" | FavoriteTargetType>("");
  const [keyword, setKeyword] = useState("");
  const [pageNum, setPageNum] = useState(1);
  const [hasMore, setHasMore] = useState(true);

  // 加载收藏列表
  const loadFavorites = useCallback(
    async (reset = false) => {
      setLoading(true);
      try {
        const query: Parameters<typeof FavoriteAPI.getPage>[0] = {
          pageNum: reset ? 1 : pageNum + 1,
          pageSize: 20,
          keywords: keyword || undefined,
          targetType: typeFilter || undefined,
        };
        const res = await FavoriteAPI.getPage(query);
        const list = (res.list as unknown as FavoriteItem[]) || [];
        if (reset) {
          setFavorites(list);
        } else {
          setFavorites((prev) => [...prev, ...list]);
        }
        setPageNum(reset ? 1 : pageNum + 1);
        setHasMore(list.length < (res.total || 0));
      } catch {
        Taro.showToast({ title: "加载失败", icon: "none" });
      } finally {
        setLoading(false);
      }
    },
    [pageNum, typeFilter, keyword]
  );

  useEffect(() => {
    loadFavorites(true);
  }, [loadFavorites]);

  // 取消收藏
  const handleUnfavorite = async (item: FavoriteItem) => {
    try {
      await FavoriteAPI.deleteByIds([item.id]);
      setFavorites((prev) => prev.filter((f) => f.id !== item.id));
      Taro.showToast({ title: "已取消收藏", icon: "success" });
    } catch {
      Taro.showToast({ title: "取消失败", icon: "none" });
    }
  };

  // 跳转处理
  const handleNavigate = (targetType: FavoriteTargetType, targetId: number) => {
    switch (targetType) {
      case "algorithm":
        Taro.navigateTo({ url: `/pages/algorithm/detail?id=${targetId}` });
        break;
      case "dataset":
        Taro.navigateTo({ url: `/pages/dataset/index?parentId=${targetId}` });
        break;
      case "image":
        // TODO: 跳转到图片详情页
        Taro.showToast({ title: "图片详情开发中", icon: "none" });
        break;
      case "result":
        Taro.showToast({ title: "结果详情开发中", icon: "none" });
        break;
      case "preset":
        Taro.showToast({ title: "预设详情开发中", icon: "none" });
        break;
    }
  };

  // 搜索处理
  const handleSearch = (value: string) => {
    setKeyword(value);
    setPageNum(1);
    loadFavorites(true);
  };

  // 清除搜索
  const handleClearSearch = () => {
    setKeyword("");
    setPageNum(1);
    loadFavorites(true);
  };

  // 类型切换
  const handleTypeChange = (key: string) => {
    setTypeFilter(key as "" | FavoriteTargetType);
    setPageNum(1);
    loadFavorites(true);
  };

  return (
    <PageLayout
      showTabbar
      currentRoute="/pages/favorite/index"
      title="我的收藏"
    >
      <View className="favorite-page">
        {/* 搜索栏 */}
        <View className="search-section">
          <SearchBar
            placeholder="搜索收藏内容..."
            value={keyword}
            onSearch={handleSearch}
            onClear={handleClearSearch}
          />
        </View>

        {/* 类型筛选 */}
        <ScrollView scrollX className="type-bar" showScrollbar={false}>
          {ALL_TYPES.map((t) => (
            <View
              key={t.key}
              className={`type-item ${typeFilter === t.key ? "active" : ""}`}
              onClick={() => handleTypeChange(t.key)}
            >
              <Text>{t.label}</Text>
            </View>
          ))}
        </ScrollView>

        {/* 收藏列表 */}
        <ScrollView scrollY className="fav-list" enhanced showScrollbar={false}>
          {loading && favorites.length === 0 ? (
            <View className="loading-wrapper">
              <View className="loading-text">加载中...</View>
            </View>
          ) : favorites.length === 0 ? (
            <View className="empty-wrapper">
              <EmptyState
                type="search"
                title="暂无收藏"
                description="收藏的内容会显示在这里"
              />
            </View>
          ) : (
            <>
              {favorites.map((item) => {
                const typeCfg =
                  TYPE_CONFIG[item.targetType as FavoriteTargetType];
                return (
                  <View key={item.id} className="fav-card">
                    {/* 缩略图 */}
                    <View
                      className="fav-thumb"
                      onClick={() =>
                        handleNavigate(
                          item.targetType as FavoriteTargetType,
                          item.targetId
                        )
                      }
                    >
                      {item.targetThumbnail ? (
                        <Image
                          src={item.targetThumbnail}
                          mode="aspectFill"
                          className="fav-thumb-img"
                        />
                      ) : (
                        <View className="fav-thumb-placeholder">
                          <Star size="20" color="#f59e0b" />
                        </View>
                      )}
                    </View>

                    {/* 信息 */}
                    <View className="fav-info">
                      <View className="fav-header">
                        <Text
                          className="fav-name"
                          onClick={() =>
                            handleNavigate(
                              item.targetType as FavoriteTargetType,
                              item.targetId
                            )
                          }
                        >
                          {item.targetName || "未命名"}
                        </Text>
                        <StatusTag
                          status={(item as any).status ?? 1}
                          size="small"
                        />
                      </View>

                      {item.targetSummary && (
                        <Text className="fav-summary" numberOfLines={2}>
                          {item.targetSummary}
                        </Text>
                      )}

                      <View className="fav-footer">
                        <View className="fav-tags">
                          {typeCfg && (
                            <View
                              className="fav-tag"
                              style={{ backgroundColor: `${typeCfg.color}20` }}
                            >
                              <Text
                                className="fav-tag-text"
                                style={{ color: typeCfg.color }}
                              >
                                {typeCfg.label}
                              </Text>
                            </View>
                          )}
                        </View>
                        <View className="fav-actions">
                          <FavoriteButton
                            targetType={item.targetType as FavoriteTargetType}
                            targetId={item.targetId}
                            size={16}
                          />
                          <View
                            className="fav-unfav-btn"
                            onClick={() => handleUnfavorite(item)}
                          >
                            <Text className="fav-unfav-text">取消</Text>
                          </View>
                        </View>
                      </View>
                    </View>
                  </View>
                );
              })}

              {!hasMore && favorites.length > 0 && (
                <View className="no-more">没有更多了</View>
              )}
              {hasMore && (
                <View className="load-more" onClick={() => loadFavorites()}>
                  <Text>加载更多</Text>
                </View>
              )}
            </>
          )}
        </ScrollView>
      </View>
    </PageLayout>
  );
};

export default FavoritePage;
