import { PageQuery } from "@/types";

/** 收藏对象类型 */
export type FavoriteTargetType = "algorithm" | "result" | "dataset" | "image" | "preset";

/** 收藏排序字段 */
export type FavoriteSortBy = "createTime";

/** 收藏查询参数 */
export interface FavoriteQuery extends PageQuery {
  /** 收藏对象类型筛选 */
  targetType?: FavoriteTargetType;
  /** 关键词搜索（按收藏对象名称） */
  keywords?: string;
  /** 排序字段：收藏时间 */
  sortBy?: FavoriteSortBy;
  /** 排序方向 */
  sortOrder?: "asc" | "desc";
}

/** 添加收藏表单 */
export interface FavoriteForm {
  /** 收藏对象类型 */
  targetType: FavoriteTargetType;
  /** 收藏对象 ID */
  targetId: number;
}

/** 收藏记录视图对象 */
export interface FavoriteVO {
  /** 收藏记录 ID */
  id: number;
  /** 用户 ID */
  userId: number;
  /** 收藏对象类型 */
  targetType: FavoriteTargetType;
  /** 收藏对象 ID */
  targetId: number;
  /** 收藏对象名称（关联查询） */
  targetName?: string;
  /** 对象摘要 */
  targetSummary?: string;
  /** 缩略图 URL */
  targetThumbnail?: string;
  /** 是否已失效（对象被删除） */
  isInvalid?: boolean;
  /** 收藏时间 */
  createTime: string;
}

/** 收藏状态（用于前端图标状态判断） */
export interface FavoriteStatus {
  /** 收藏对象类型 */
  targetType: FavoriteTargetType;
  /** 收藏对象 ID */
  targetId: number;
  /** 是否已收藏 */
  favorited: boolean;
}

/** 收藏数量统计（按类型分组） */
export interface FavoriteCount {
  /** 收藏对象类型 */
  targetType: FavoriteTargetType;
  /** 该类型收藏数量 */
  count: number;
}
