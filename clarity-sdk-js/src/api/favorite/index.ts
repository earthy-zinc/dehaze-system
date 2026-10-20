import { PageResult } from "@/types";
import request from "@/utils/request";
import {
  FavoriteCount,
  FavoriteForm,
  FavoriteQuery,
  FavoriteStatus,
  FavoriteTargetType,
  FavoriteVO,
} from "./model";

/**
 * 收藏管理 API
 * 为算法、处理结果、数据集等业务实体提供统一的收藏能力。
 */
class FavoriteAPI {
  /** 收藏列表分页查询（支持类型筛选、排序、关键词搜索） */
  static getPage(query?: FavoriteQuery) {
    return request<PageResult<FavoriteVO[]>>({
      url: "/api/v1/favorites/page",
      method: "get",
      params: query,
    });
  }

  /** 添加收藏（同一用户对同一对象只能收藏一次） */
  static add(data: FavoriteForm) {
    return request<number>({
      url: "/api/v1/favorites",
      method: "post",
      data,
    });
  }

  /** 批量取消收藏 */
  static deleteByIds(ids: number[]) {
    if (!ids || ids.length === 0) {
      return Promise.reject(new Error("待取消收藏的 ID 列表不能为空"));
    }
    return request({
      url: "/api/v1/favorites/" + ids.join(","),
      method: "delete",
    });
  }

  /** 检查指定对象是否已收藏（用于前端图标状态判断） */
  static getStatus(targetType: FavoriteTargetType, targetId: number) {
    return request<FavoriteStatus>({
      url: `/api/v1/favorites/${targetId}/status`,
      method: "get",
      params: { targetType },
    });
  }

  /** 收藏数量统计（按类型分组） */
  static getCount(targetType?: FavoriteTargetType) {
    return request<FavoriteCount[]>({
      url: "/api/v1/favorites/count",
      method: "get",
      params: targetType ? { targetType } : undefined,
    });
  }
}

export default FavoriteAPI;
