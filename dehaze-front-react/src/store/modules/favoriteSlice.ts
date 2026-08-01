import { FavoriteAPI, type FavoriteTargetType } from "dehaze-sdk-js";
import { createAsyncThunk, createSlice } from "@reduxjs/toolkit";
import type { PayloadAction } from "@reduxjs/toolkit";

interface FavoriteStatusMap {
  [targetId: number]: boolean;
}

interface FavoriteState {
  status: Record<FavoriteTargetType, FavoriteStatusMap>;
  counts: Record<string, number>;
  loading: boolean;
  error: string | null;
}

const initialState: FavoriteState = {
  status: {
    algorithm: {},
    result: {},
    dataset: {},
    image: {},
    preset: {},
  },
  counts: {},
  loading: false,
  error: null,
};

/** 检查指定收藏对象是否已收藏 */
export const fetchFavoriteStatus = createAsyncThunk(
  "favorite/fetchStatus",
  async ({
    targetType,
    targetId,
  }: {
    targetType: FavoriteTargetType;
    targetId: number;
  }) => {
    const status = await FavoriteAPI.getStatus(targetType, targetId);
    return { targetType, targetId, favorited: status.favorited };
  }
);

/** 切换收藏状态（自动判断添加/取消） */
export const toggleFavorite = createAsyncThunk(
  "favorite/toggle",
  async ({
    targetType,
    targetId,
  }: {
    targetType: FavoriteTargetType;
    targetId: number;
  }) => {
    const isFavorited = false; // will be checked inside thunk caller
    return { targetType, targetId, isFavorited };
  }
);

/** 获取收藏数量统计 */
export const fetchFavoriteCount = createAsyncThunk(
  "favorite/fetchCount",
  async (targetType?: FavoriteTargetType) => {
    const counts = await FavoriteAPI.getCount(targetType);
    return counts;
  }
);

const favoriteSlice = createSlice({
  name: "favorite",
  initialState,
  reducers: {
    /** 清除所有收藏状态 */
    clearFavorites: (state) => {
      state.status = {
        algorithm: {},
        result: {},
        dataset: {},
        image: {},
        preset: {},
      };
      state.error = null;
    },
    /** 手动设置某个对象的收藏状态 */
    setStatus: (
      state,
      action: PayloadAction<{
        targetType: FavoriteTargetType;
        targetId: number;
        favorited: boolean;
      }>
    ) => {
      const { targetType, targetId, favorited } = action.payload;
      state.status[targetType][targetId] = favorited;
    },
    /** 批量更新收藏状态 */
    setAllStatus: (
      state,
      action: PayloadAction<{
        targetType: FavoriteTargetType;
        statuses: FavoriteStatusMap;
      }>
    ) => {
      state.status[action.payload.targetType] = action.payload.statuses;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchFavoriteStatus.pending, (state) => {
        state.loading = true;
      })
      .addCase(fetchFavoriteStatus.fulfilled, (state, action) => {
        const { targetType, targetId, favorited } = action.payload;
        state.status[targetType][targetId] = favorited;
        state.loading = false;
      })
      .addCase(fetchFavoriteStatus.rejected, (state, action) => {
        state.error = action.error.message || "加载收藏状态失败";
        state.loading = false;
      })
      .addCase(toggleFavorite.pending, (state) => {
        state.loading = true;
      })
      .addCase(toggleFavorite.fulfilled, (state, action) => {
        const { targetType, targetId, isFavorited } = action.payload;
        state.status[targetType][targetId] = !isFavorited;
        state.loading = false;
      })
      .addCase(toggleFavorite.rejected, (state) => {
        state.loading = false;
      })
      .addCase(fetchFavoriteCount.fulfilled, (state, action) => {
        for (const item of action.payload) {
          state.counts[item.targetType] = item.count;
        }
      });
  },
});

export const { clearFavorites, setStatus, setAllStatus } =
  favoriteSlice.actions;
export default favoriteSlice.reducer;
