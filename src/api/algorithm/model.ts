/**
 * 数据集查询参数类型
 */
export interface AlgorithmQuery {
  keywords?: string;
}

export interface Algorithm {
  id: number;
  name: string;
  type: string;
  description: string;
  importPath: string;
  children: Model[];
}

interface ModelWithPath {
  id: number;
  name: string;
  type: string;
  description: string;
  path: string;
  /**
   * 状态(1:启用；0:禁用)
   */
  status?: number;
  size: number;
  children?: never; // 确保 children 不被定义
}

interface ModelWithChildren {
  id: number;
  name: string;
  type: string;
  description: string;
  path?: never; // 确保 path 不被定义
  children: ModelWithChildren[]; // 注意这里的类型应该是自身接口的数组，形成递归
}

export type Model = ModelWithPath | ModelWithChildren;
