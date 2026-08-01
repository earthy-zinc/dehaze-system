import { FavoriteAPI, FavoriteTargetType } from "dehaze-sdk-js";
import Taro from "@tarojs/taro";
import { useState, useCallback, useEffect } from "react";

/**
 * 收藏状态切换 Hook
 * @param targetType 收藏对象类型
 * @param targetId 收藏对象 ID
 */
export function useFavorite(targetType: FavoriteTargetType, targetId: number) {
  const [isFavorited, setIsFavorited] = useState(false);
  const [loading, setLoading] = useState(false);

  const checkStatus = useCallback(async () => {
    if (!targetId) return;
    try {
      const status = await FavoriteAPI.getStatus(targetType, targetId);
      setIsFavorited(status.favorited);
    } catch {
      // 静默失败
    }
  }, [targetType, targetId]);

  useEffect(() => {
    checkStatus();
  }, [checkStatus]);

  const toggle = useCallback(async () => {
    if (loading) return;
    setLoading(true);
    try {
      if (isFavorited) {
        const result = await FavoriteAPI.getPage({
          targetType,
          keywords: "",
          pageNum: 1,
          pageSize: 1,
        });
        if (result.list && result.list.length > 0) {
          await FavoriteAPI.deleteByIds([result.list[0].id]);
          setIsFavorited(false);
          Taro.showToast({ title: "已取消收藏", icon: "none" });
        }
      } else {
        await FavoriteAPI.add({ targetType, targetId });
        setIsFavorited(true);
        Taro.showToast({ title: "收藏成功", icon: "success" });
      }
    } catch {
      Taro.showToast({ title: "操作失败", icon: "none" });
    } finally {
      setLoading(false);
    }
  }, [targetType, targetId, isFavorited, loading]);

  return { isFavorited, loading, toggle };
}
