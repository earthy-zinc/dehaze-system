import { PageQuery } from "@/types";

/** 历史记录创建表单 */
export interface HistoryForm {
  originalImageUrl?: string;
  originalThumbnailUrl?: string;
  resultImageUrl?: string;
  resultThumbnailUrl?: string;
  /** 图像模型标识（统一模型规范 slug） */
  modelId?: string;
  modelName?: string;
  modelParams?: string;
  /** 处理耗时（毫秒）；由图像任务的 time（秒）写入时需 ×1000 */
  processingTime?: number;
  status?: number;
  inputSource?: string;
}

/** 历史记录查询参数（收藏状态由收藏管理模块 /api/v1/favorites 维护，不在此接口） */
export interface HistoryQuery extends PageQuery {
  status?: number;
  inputSource?: string;
  /** 关键词（模糊匹配模型名称/原图地址） */
  keywords?: string;
}

/** 历史记录视图对象 */
export interface InputHistoryVO {
  id: number;
  originalImageUrl?: string;
  originalThumbnailUrl?: string;
  resultImageUrl?: string;
  resultThumbnailUrl?: string;
  /** 图像模型标识（统一模型规范 slug） */
  modelId?: string;
  modelName?: string;
  modelParams?: string;
  /** 处理耗时（毫秒）；与图像任务/评估的 time（秒）不同单位 */
  processingTime?: number;
  status?: number;
  inputSource?: string;
  createTime?: string;
}
