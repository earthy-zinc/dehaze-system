import { FavoriteForm, FavoriteQuery, FavoriteTargetType } from "@/api/favorite/model";
import { pageQuery } from "./common";

/** 创建添加收藏表单 */
export function createFavoriteForm(overrides: Partial<FavoriteForm> = {}): FavoriteForm {
  return {
    targetType: "algorithm",
    targetId: 1,
    ...overrides,
  };
}

/** 创建收藏查询参数 */
export function createFavoriteQuery(overrides: Partial<FavoriteQuery> = {}): FavoriteQuery {
  return pageQuery<FavoriteQuery>({
    pageNum: 1,
    pageSize: 10,
    ...overrides,
  });
}

/** 所有收藏对象类型（用于遍历测试） */
export const ALL_TARGET_TYPES: FavoriteTargetType[] = [
  "algorithm",
  "result",
  "dataset",
  "image",
  "preset",
];
