/**
 * 模型预测参数
 * 后端流程：
 * 1. 根据id获取模型import路径、文件存储路径
 * 2. 根据模型import路径导入模型
 * 3. 根据模型文件存储路径获取模型文件
 * 4. 根据模型文件预测
 */
export interface PredParam {
  modelId: number;
  url: string;
  modelParam?: Object;
}

export interface PredResult {
  predUrl: string;
  hazeUrl: string;
}

export interface EvalParam {
  modelId: number;
  predUrl: string;
  gtUrl?: string;
}

export interface EvalResult {
  id: number;
  // 评价指标的名称
  label: string;
  // 评价指标的值
  value: string;
  // 基准值
  baseline?: string;
  // 评价指标是越高越好还是越低越好
  better?: "higher" | "lower";
  // 评价指标的描述
  description?: string;
}

export interface PredImageInfo {
  name: string;
  path: string;
  url: string;
}
