/** 模型查询参数类型 */
export interface AlgorithmQuery {
  keywords?: string;
}

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
