import { PageQuery } from "@/types";

/** 历史记录创建表单 */
export interface HistoryForm {
  originalImageUrl?: string;
  originalThumbnailUrl?: string;
  resultImageUrl?: string;
  resultThumbnailUrl?: string;
  algorithmId?: number;
  algorithmName?: string;
  algorithmParams?: string;
  processingTime?: number;
  status?: number;
  inputSource?: string;
}

/** 历史记录更新表单 */
export interface HistoryUpdateForm {}

/** 历史记录查询参数 */
export interface HistoryQuery extends PageQuery {
  status?: number;
  inputSource?: string;
}

/** 历史记录视图对象 */
export interface InputHistoryVO {
  id: number;
  originalImageUrl?: string;
  originalThumbnailUrl?: string;
  resultImageUrl?: string;
  resultThumbnailUrl?: string;
  algorithmId?: number;
  algorithmName?: string;
  algorithmParams?: string;
  processingTime?: number;
  status?: number;
  inputSource?: string;
  syncStatus?: number;
  createTime?: string;
}
