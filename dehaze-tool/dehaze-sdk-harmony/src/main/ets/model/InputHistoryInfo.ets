/**
 * 图像输入历史记录数据模型
 *
 * 与后端 model.SysInputHistory 对齐
 */

/** 历史记录处理状态 */
export enum HistoryStatus {
  /** 成功 */
  SUCCESS = 1,
  /** 失败 */
  FAILED = 2,
  /** 处理中 */
  PROCESSING = 3,
}

/** 历史记录查询参数 */
export class InputHistoryQuery {
  pageNum?: number = 1;
  pageSize?: number = 10;
  /** 仅查询收藏 */
  isFavorite?: boolean = undefined;
}

/**
 * 图像输入历史记录（后端 model.SysInputHistory）
 */
export class InputHistory {
  id: number = 0;
  /** 用户 ID */
  userId: number = 0;
  /** 原始图片 URL */
  originalImageUrl?: string = undefined;
  /** 原始缩略图 URL */
  originalThumbnailUrl?: string = undefined;
  /** 处理结果图片 URL */
  resultImageUrl?: string = undefined;
  /** 结果缩略图 URL */
  resultThumbnailUrl?: string = undefined;
  /** 算法 ID */
  algorithmId?: number = undefined;
  /** 算法名称（冗余） */
  algorithmName?: string = undefined;
  /** 算法参数（JSON） */
  algorithmParams?: string = undefined;
  /** 处理耗时（毫秒） */
  processingTime?: number = undefined;
  /** 处理状态（1=成功，2=失败，3=处理中） */
  status?: number = undefined;
  /** 图片来源（upload/camera/sample） */
  inputSource?: string = undefined;
  /** 是否收藏 */
  isFavorite?: boolean = undefined;
  createTime?: string = undefined;
  updateTime?: string = undefined;
}

/** 创建历史记录请求 */
export class CreateInputHistory {
  originalImageUrl: string = '';
  originalThumbnailUrl?: string = undefined;
  resultImageUrl?: string = undefined;
  resultThumbnailUrl?: string = undefined;
  algorithmId?: number = undefined;
  algorithmName?: string = undefined;
  algorithmParams?: string = undefined;
  processingTime?: number = undefined;
  status: number = 3;
  inputSource: string = 'upload';
}
