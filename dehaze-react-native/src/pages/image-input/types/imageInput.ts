/**
 * 图像输入模块类型定义
 */

import type { InputHistoryVO } from 'dehaze-sdk-js';

/** 输入方式枚举 */
export type InputMethod = 'upload' | 'camera' | 'sample' | 'history';

/** 样例图片分类（与后端 hazeLevel 对齐） */
export type SampleCategory = 'all' | 'light' | 'medium' | 'heavy';

/** 难度等级 */
export type DifficultyLevel = 'easy' | 'medium' | 'hard';

/** 样例图片模型（由后端 DatasetItemVO 的 hazyImages 映射而来） */
export interface SampleImage {
  id: number;
  name: string;
  url: string;
  thumbUrl?: string;
  category: SampleCategory;
  difficulty: DifficultyLevel;
  sceneType?: string;
  width?: number;
  height?: number;
}

/** 历史记录（直接复用 SDK 的 InputHistoryVO） */
export type HistoryRecord = InputHistoryVO;

/** 输入方式配置 */
export interface InputMethodConfig {
  key: InputMethod;
  icon: string;
  title: string;
  subtitle: string;
}

/** 图片验证结果 */
export interface ValidationResult {
  valid: boolean;
  error?: string;
}

/** 图片信息 */
export interface ImageInfo {
  width: number;
  height: number;
  fileSize: number;
  type: string;
}

/** 分类标签配置 */
export interface CategoryConfig {
  key: SampleCategory;
  label: string;
}

/** 历史记录分组 */
export interface HistoryGroup {
  title: string;
  data: HistoryRecord[];
}
