/**
 * 模型预测参数
 * 后端流程：
 * 1. 根据id获取模型import路径、文件存储路径
 * 2. 根据模型import路径导入模型
 * 3. 根据模型文件存储路径获取模型文件
 * 4. 根据模型文件预测
 */
export class PredParam {
  modelId: number = 0;
  url: string = '';
  modelParam?: Object = undefined;
}

export class PredResult {
  predUrl: string = '';
  hazeUrl: string = '';
}

export class EvalParam {
  modelId: number = 0;
  predUrl: string = '';
  gtUrl?: string = undefined;
}

export class EvalResult {
  id: number = 0;
  // 评价指标的名称
  label: string = '';
  // 评价指标的值
  value: string = '';
  // 基准值
  baseline?: string = undefined;
  // 评价指标是越高越好还是越低越好
  better?: string = undefined;
  // 评价指标的描述
  description?: string = undefined;
}

export class PredImageInfo {
  name: string = '';
  path: string = '';
  url: string = '';
}