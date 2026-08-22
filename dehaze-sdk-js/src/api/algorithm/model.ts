/** 算法查询参数 */
export interface AlgorithmQuery {
  keywords?: string;
}

/** 算法视图对象（新增/修改时兼作 AlgorithmForm 提交体） */
export interface Algorithm {
  id: number;
  parentId: number;
  name: string;
  type: string;
  description: string;
  img?: string;
  path?: string;
  importPath?: string;
  params?: string;
  flops?: string;
  status?: number;
  size?: string;
  version?: string;
  auditBy?: number;
  auditTime?: string;
  auditRemark?: string;
  createTime?: string;
  children?: Algorithm[];
}

type Optional<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;

export type CreateAlgorithmOptional = Optional<Algorithm, "id" | "description">;

/** 算法审核表单 */
export interface AlgorithmAuditForm {
  approved: boolean;
  remark?: string;
}

/** 算法版本创建表单 */
export interface AlgorithmVersionForm {
  version: string;
  changeLog?: string;
  modelFileId?: number;
}

/** 算法版本视图对象 */
export interface AlgorithmVersionVO {
  id: number;
  algorithmId: number;
  version: string;
  changeLog?: string;
  status?: number;
  isActive?: boolean;
  modelFileId?: number;
  createTime?: string;
}

/** 算法监控数据 */
export interface AlgorithmMonitorVO {
  callCount: number;
  avgTime: number;
  successRate: number;
  todayCallCount: number;
}

/** 算法监控统计报表条目（按日期聚合） */
export interface AlgorithmMonitorStatsItemVO {
  date: string;
  callCount: number;
  avgTime: number;
  successRate: number;
}

/** 算法对比表单 */
export interface AlgorithmCompareForm {
  /** 算法ID列表（2-3 个） */
  algorithmIds: number[];
  /** 文件ID（与 imageUrl 二选一） */
  fileId?: number;
  /** 图片 URL（与 fileId 二选一） */
  imageUrl?: string;
}

/** 算法对比结果项 */
export interface AlgorithmCompareVO {
  algorithmId: number;
  algorithmName: string;
  /** 处理结果 URL（服务端已执行对比预测时返回） */
  resultUrl?: string;
  /** 处理耗时（毫秒） */
  time?: number;
  /** 评估指标（PSNR/SSIM 等，JSON 字符串） */
  metrics?: string;
}

/** 算法选择树节点 */
export interface AlgorithmSelectNodeVO {
  id: number;
  /** 父节点 ID（根节点为 0） */
  parentId: number;
  name: string;
  /** 算法类型 */
  type: string;
  /** 是否为叶子节点（算法节点） */
  isLeaf: boolean;
  children?: AlgorithmSelectNodeVO[];
}

/** 算法详情 */
export interface AlgorithmDetailVO {
  id: number;
  name: string;
  type: string;
  /** 算法图片 */
  img?: string;
  description: string;
  path?: string;
  /** 模型文件大小 */
  size?: string;
  /** 参数量 */
  params?: string;
  /** FLOPs */
  flops?: string;
  /** 算法版本 */
  version?: string;
  /** 算法状态 */
  status?: number;
  /** 平均评分 */
  avgRating?: number;
  /** 评价总数 */
  ratingCount?: number;
  /** 使用次数 */
  usageCount?: number;
  /** 样例效果图 URL 列表 */
  sampleImages?: string[];
}

/** 自定义图片测试表单 */
export interface AlgorithmTestForm {
  /** 文件 ID（与 imageUrl 二选一） */
  fileId?: number;
  /** 图片 URL（与 fileId 二选一） */
  imageUrl?: string;
  /** 预测参数（JSON） */
  params?: string;
}

/** 算法推荐匹配表单 */
export interface AlgorithmRecommendForm {
  keyword?: string;
  taskType?: string;
  sampleAlgorithmId?: number;
  topN?: number;
}

/** 算法推荐匹配结果项 */
export interface AlgorithmRecommendVO {
  algorithmId: number;
  algorithmName: string;
  matchScore: number;
  reason: string;
  estimatedTime?: number;
}

/** 算法推荐匹配结果 */
export interface AlgorithmRecommendResult {
  total: number;
  items: AlgorithmRecommendVO[];
}
