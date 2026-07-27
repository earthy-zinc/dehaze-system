import { PageQuery } from "@/types";

// ==================== 数据集相关类型 ====================

/**
 * 数据集查询参数
 */
export interface DatasetQuery extends PageQuery {
  keyword?: string;
  type?: string;
  status?: string;
}

/**
 * 数据集创建表单
 */
export interface DatasetAddForm {
  /** 父数据集ID，0表示根数据集 */
  parentId: number;
  /** 数据集类型 */
  type?: string;
  /** 数据集名称 */
  name?: string;
  /** 数据集描述信息 */
  description?: string;
  /** 数据集存储路径 */
  path?: string;
  /** 数据集状态：1-启用，0-禁用 */
  status?: string;
}

/**
 * 数据集更新表单
 */
export interface DatasetUpdateForm {
  /** 数据集类型 */
  type?: string;
  /** 数据集名称 */
  name?: string;
  /** 数据集描述信息 */
  description?: string;
  /** 数据集存储路径 */
  path?: string;
  /** 数据集状态：1-启用，0-禁用 */
  status?: string;
}

/**
 * 数据集统计信息
 */
export interface DatasetStatistics {
  /** 数据项总数 */
  itemCount: number;
  /** 文件总数 */
  fileCount: number;
  /** 总大小（字节） */
  totalSize: number;
  /** 已标注图片数量（haze_level 非空） */
  annotatedCount: number;
  /** 未标注图片数量（haze_level 为空） */
  unannotatedCount: number;
  /** 场景类型分布 */
  sceneDistribution: Record<string, number>;
  /** 雾霾程度分布 */
  hazeDistribution: Record<string, number>;
  /** 文件格式分布 */
  formatDistribution: Record<string, number>;
  /** 分辨率分布 */
  resolutionDistribution?: Record<string, number>;
}

/**
 * 数据集视图对象
 */
export interface Dataset {
  /** 数据集ID */
  id: number;
  /** 父数据集ID，null表示根数据集 */
  parentId?: number;
  /** 数据集类型：training, test, user, result */
  type: string;
  /** 数据集名称 */
  name: string;
  /** 数据集描述信息 */
  description?: string;
  /** 数据集存储路径 */
  path: string;
  /** 是否有子数据集 */
  hasChildren?: boolean;
  /** 子数据集列表 */
  children?: Dataset[];
  /** 数据集状态：1-启用，0-禁用 */
  status?: number;
  /** 统计信息 */
  statistics?: DatasetStatistics;
  /** 图片总数（用于列表展示） */
  total?: number;
  /** 数据集创建时间 */
  createTime?: Date | string;
  /** 数据集最后修改时间 */
  updateTime?: Date | string;
}

// ==================== 数据项相关类型 ====================

/**
 * 数据项查询参数
 */
export interface DatasetItemQuery extends PageQuery {
  /** 数据集ID */
  datasetId?: number;
  /** 搜索关键词 */
  keyword?: string;
  /** 场景类型 */
  sceneType?: string;
  /** 雾霾程度，支持多种规范：light/medium/heavy、beta=0.5、A=0.8,beta=0.2 等，不硬性枚举校验 */
  hazeLevel?: string;
  /** 最小图片宽度 */
  minWidth?: number;
  /** 最大图片宽度 */
  maxWidth?: number;
  /** 最小图片高度 */
  minHeight?: number;
  /** 最大图片高度 */
  maxHeight?: number;
  /** 最小文件大小 */
  minSize?: number;
  /** 最大文件大小 */
  maxSize?: number;
  /** 排序字段：relevance, createTime, usageCount */
  sortBy?: "relevance" | "createTime" | "usageCount";
  /** 排序方向：asc, desc */
  sortOrder?: "asc" | "desc";
}

/**
 * 数据项创建表单
 */
export interface DatasetItemCreateForm {
  /** 所属数据集ID */
  datasetId: number;
  /** 数据项名称 */
  name?: string;
  /** 场景类型 */
  sceneType?: string;
  /** 数据项描述信息 */
  description?: string;
}

/**
 * 数据项更新表单
 */
export interface DatasetItemUpdateForm {
  /** 数据项名称 */
  name?: string;
  /** 场景类型 */
  sceneType?: string;
}

/**
 * 数据项上传表单
 */
export interface DatasetItemUploadForm {
  /** 数据集ID */
  datasetId: number;
  /** 数据项名称 */
  name?: string;
  /** 清晰图文件（base64或二进制），可选（适配不同数据集规范） */
  clearImage?: string | Blob;
  /** 有雾图文件列表，可选（适配不同数据集规范） */
  hazyImages?: (string | Blob)[];
  /** 对应的雾霾程度列表，支持多种规范：light/medium/heavy、beta=0.5、空字符串等 */
  hazeLevels?: string[];
  /** 场景类型 */
  sceneType?: string;
}

/**
 * 批量数据项上传表单
 */
export interface BatchDatasetItemUploadForm {
  /** 数据集ID */
  datasetId: number;
  /** 文件列表 */
  files: (string | Blob)[];
  /** 场景类型（可选，应用于所有配对） */
  sceneType?: string;
}

/**
 * 数据项简要视图对象
 */
export interface DatasetItemSimpleVO {
  /** 数据项ID */
  id: number;
  /** 所属数据集ID */
  datasetId: number;
  /** 数据项名称 */
  name: string;
  /** 场景类型 */
  sceneType?: string;
  /** 数据项描述信息 */
  description?: string;
}

/**
 * 数据项详情视图对象
 */
export interface DatasetItemVO {
  /** 数据项ID */
  id: number;
  /** 所属数据集ID */
  datasetId: number;
  /** 数据项名称 */
  name: string;
  /** 场景类型 */
  sceneType?: string;
  /** 数据项描述信息 */
  description?: string;
  /** 使用次数 */
  usageCount?: number;
  /** 图片总数 */
  imageCount?: number;
  /** 清晰图信息 */
  clearImage?: ImageUrlVO;
  /** 有雾图列表 */
  hazyImages?: ImageUrlVO[];
  /** 数据项创建时间 */
  createTime?: Date | string;
  /** 数据项最后更新时间 */
  updateTime?: Date | string;
}

// ==================== 图片文件相关类型 ====================

/**
 * 图片上传表单
 */
export interface ItemFileUploadForm {
  /** 图片文件（base64或二进制） */
  file: string | Blob;
  /** 所属数据项ID */
  itemId: number;
  /** 图片类型：clear-清晰图，hazy-有雾图，trans-透射图，depth-深度图，segment-分割图 */
  type: "clear" | "hazy" | "trans" | "depth" | "segment";
  /** 图片描述信息 */
  description?: string;
  /** 场景类型 */
  sceneType?: string;
  /** 雾霾程度，支持多种规范：light/medium/heavy、beta=0.5、A=0.8,beta=0.2 等，可为空 */
  hazeLevel?: string;
}

/**
 * 图片更新表单
 */
export interface ItemFileUpdateForm {
  /** 图片类型：clear-清晰图，hazy-有雾图，trans-透射图，depth-深度图，segment-分割图 */
  type?: "clear" | "hazy" | "trans" | "depth" | "segment";
  /** 场景类型 */
  sceneType?: string;
  /** 雾霾程度，支持多种规范：light/medium/heavy、beta=0.5、A=0.8,beta=0.2 等，可为空 */
  hazeLevel?: string;
  /** 图片描述信息 */
  description?: string;
}

/**
 * 图片简要视图对象
 */
export interface SimpleImageUrlVO {
  /** 数据项文件ID */
  id: number;
  /** 所属数据项ID */
  itemId: number;
  /** 所属数据集ID */
  datasetId: number;
  /** 图片类型：clear-清晰图，hazy-有雾图，trans-透射图，depth-深度图，segment-分割图 */
  type: string;
  /** 图片访问URL */
  url: string;
  /** 缩略图URL */
  thumbnailUrl?: string;
  /** 图片描述信息 */
  description?: string;
  /** 图片宽度 */
  width?: number;
  /** 图片高度 */
  height?: number;
  /** 雾霾程度，支持多种规范：light/medium/heavy、beta=0.5、A=0.8,beta=0.2 等，可为空 */
  hazeLevel?: string;
  /** 文件名 */
  fileName?: string;
  /** 文件大小（字节） */
  sizeBytes?: number;
  /** 文件大小，格式化显示 */
  formattedSize?: string;
  /** 文件格式 */
  format?: string;
  /** 图片创建时间 */
  createTime?: Date | string;
}

/**
 * 图片详情视图对象
 */
export interface ImageUrlVO {
  /** 数据项文件ID */
  id: number;
  /** 所属数据项ID */
  itemId: number;
  /** 所属数据集ID */
  datasetId: number;
  /** 所属数据集名称 */
  datasetName?: string;
  /** 数据项简要信息 */
  datasetItem?: DatasetItemSimpleVO;
  /** 图片类型：clear-清晰图，hazy-有雾图，trans-透射图，depth-深度图，segment-分割图 */
  type: string;
  /** 图片访问URL */
  url: string;
  /** 原始图片URL */
  originUrl?: string;
  /** 缩略图URL */
  thumbnailUrl?: string;
  /** 图片描述信息 */
  description?: string;
  /** 图片宽度 */
  width?: number;
  /** 图片高度 */
  height?: number;
  /** 场景类型 */
  sceneType?: string;
  /** 雾霾程度，支持多种规范：light/medium/heavy、beta=0.5、A=0.8,beta=0.2 等，可为空 */
  hazeLevel?: string;
  /** 文件名 */
  fileName?: string;
  /** 文件大小（字节） */
  sizeBytes?: number;
  /** 文件大小，格式化显示 */
  formattedSize?: string;
  /** 文件格式 */
  format?: string;
  /** 文件MD5值 */
  md5?: string;
  /** 使用次数 */
  usageCount?: number;
  /** 图片上传时间 */
  createTime?: Date | string;
  /** 是否有配对图片 */
  hasPairedImages?: boolean;
  /** 配对图片列表 */
  pairedFiles?: SimpleImageUrlVO[];
  /** 配对图片总数 */
  pairedCount?: number;
}

// ==================== 批量操作相关类型 ====================

/**
 * 批量删除表单
 */
export interface BatchDeleteForm {
  /** ID列表 */
  ids: number[];
}

/**
 * 失败项详情
 */
export interface FailedItem {
  /** ID */
  id?: number;
  /** 失败原因 */
  reason: string;
}

/**
 * 批量删除结果
 */
export interface BatchDeleteResultVO {
  /** 删除成功的ID列表 */
  successIds: number[];
  /** 删除失败的详细信息 */
  failedItems: FailedItem[];
  /** 成功删除数量 */
  successCount: number;
  /** 失败删除数量 */
  failedCount: number;
}

/**
 * 批量上传成功项
 */
export interface BatchUploadSuccessItemVO {
  /** 数据项ID */
  id: number;
  /** 数据项名称 */
  name: string;
  /** 文件数量（清晰图+有雾图） */
  fileCount: number;
}

/**
 * 批量上传失败项
 */
export interface BatchUploadFailedItemVO {
  /** 文件名 */
  fileName: string;
  /** 失败原因 */
  reason: string;
}

/**
 * 批量上传结果
 */
export interface BatchUploadResultVO {
  /** 总文件数 */
  total: number;
  /** 成功数量 */
  succeeded: number;
  /** 失败数量 */
  failed: number;
  /** 成功项详情列表 */
  successItems: BatchUploadSuccessItemVO[];
  /** 失败项详情列表 */
  failedItems: BatchUploadFailedItemVO[];
}

/**
 * 批量操作失败详情
 */
export interface BatchActionFailureDetailVO {
  /** 失败记录的唯一标识 */
  identifier?: string;
  /** 失败原因 */
  reason: string;
}

/**
 * 批量操作结果
 */
export interface BatchOperationResultVO {
  /** 成功数量 */
  successCount: number;
  /** 失败数量 */
  failedCount: number;
  /** 操作消息 */
  message: string;
  /** 成功的ID列表 */
  successIds?: number[];
  /** 失败详情列表 */
  failureDetails?: BatchActionFailureDetailVO[];
}
