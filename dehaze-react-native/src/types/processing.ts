/**
 * 去雾处理模块类型定义
 *
 * 任务状态、处理阶段、参数定义、批量任务等。
 */

/** 处理任务状态 */
export type TaskStatus =
  | 'idle' // 空闲，未开始
  | 'preprocessing' // 图片预处理
  | 'initializing' // 算法初始化
  | 'processing' // 去雾处理中
  | 'postprocessing' // 后处理优化
  | 'saving' // 结果保存
  | 'success' // 处理成功
  | 'failed' // 处理失败
  | 'canceled'; // 已取消

/** 处理阶段定义（用于进度展示） */
export interface ProcessingStage {
  key: TaskStatus;
  label: string;
  /** 阶段起始进度（0-100） */
  start: number;
  /** 阶段结束进度（0-100） */
  end: number;
}

/** 单个任务进度信息 */
export interface TaskProgress {
  status: TaskStatus;
  /** 当前整体进度 0-100 */
  percent: number;
  /** 当前阶段描述 */
  stageLabel: string;
  /** 已用时间（ms） */
  elapsed: number;
  /** 错误信息 */
  error?: string;
}

/** 通用算法参数 */
export interface CommonAlgorithmParams {
  /** 去雾强度 0-100 */
  strength?: number;
  /** 色彩饱和度 0-200 */
  saturation?: number;
  /** 对比度 0-200 */
  contrast?: number;
  /** 锐化程度 0-100 */
  sharpen?: number;
}

/** 参数 schema（用于动态渲染参数调节面板） */
export interface ParamSchema {
  key: keyof CommonAlgorithmParams;
  label: string;
  type: 'slider' | 'select';
  min?: number;
  max?: number;
  step?: number;
  default: number;
  options?: { label: string; value: number }[];
  description?: string;
}

/** 单张处理结果 */
export interface ProcessingResult {
  logId?: number;
  resultUrl: string;
  resultThumbnailUrl?: string;
  /** 处理耗时（ms） */
  time: number;
  /** 是否命中缓存 */
  fromCache?: boolean;
}

/** 批量任务条目 */
export interface BatchTaskItem {
  id: string;
  image: { url: string; name?: string };
  status: TaskStatus;
  percent: number;
  /** 处理耗时（ms），完成后填充 */
  time?: number;
  error?: string;
  result?: ProcessingResult;
}

/** 预设参数方案 */
export interface ParamPreset {
  key: string;
  name: string;
  description?: string;
  params: CommonAlgorithmParams;
}
