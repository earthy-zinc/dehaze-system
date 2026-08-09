import { FavoriteTargetType } from "dehaze-sdk-js";
import { useCallback, useEffect } from "react";
import { useSelector } from "react-redux";
import { useAppDispatch, type RootState } from "@/store/hooks";
import {
  fetchFavoriteStatus,
  toggleFavorite,
  fetchFavoriteCount,
} from "@/store/modules/favoriteSlice";
import { message } from "antd";

export function useFavorite(targetType: FavoriteTargetType, targetId: number) {
  const dispatch = useAppDispatch();
  const status = useSelector(
    (state: RootState) => state.favorite.status[targetType]?.[targetId] ?? false
  );
  const loading = useSelector((state: RootState) => state.favorite.loading);

  useEffect(() => {
    dispatch(fetchFavoriteStatus({ targetType, targetId }));
  }, [dispatch, targetType, targetId]);

  const toggle = useCallback(async () => {
    try {
      const result: any = await dispatch(
        toggleFavorite({ targetType, targetId })
      ).unwrap();
      message.success(result.favorited ? "已加入收藏" : "已取消收藏");
    } catch (err: any) {
      message.error(err?.message || "操作失败");
    }
  }, [dispatch, targetType, targetId]);

  return { isFavorited: status, loading, toggle };
}

export function useFavoriteCount() {
  const dispatch = useAppDispatch();
  const counts = useSelector((state: RootState) => state.favorite.counts);

  useEffect(() => {
    dispatch(fetchFavoriteCount());
  }, [dispatch]);

  return counts;
}
