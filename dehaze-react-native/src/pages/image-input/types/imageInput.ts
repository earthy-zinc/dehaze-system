/**
 * 图像输入模块类型定义
 */

/** 输入方式枚举 */
export type InputMethod = 'upload' | 'camera' | 'sample' | 'history';

/** 样例图片分类 */
export type SampleCategory = 'all' | 'light' | 'medium' | 'heavy' | 'special';

/** 难度等级 */
export type DifficultyLevel = 'easy' | 'medium' | 'hard';

/** 图片来源 */
export type ImageSource = 'upload' | 'camera' | 'sample' | 'history';

/** 选中的图片模型 */
export interface SelectedImage {
  id: string;
  uri: string;
  filename: string;
  width: number;
  height: number;
  fileSize: number;
  source: ImageSource;
  sampleInfo?: SampleImage;
}

/** 样例图片模型 */
export interface SampleImage {
  id: number;
  name: string;
  url: string;
  category: SampleCategory;
  difficulty: DifficultyLevel;
  sceneType: string;
  recommendedAlgorithm?: string;
}

/** 历史记录模型 */
export interface HistoryRecord {
  id: string;
  originalThumbnail: string;
  resultThumbnail?: string;
  filename: string;
  timestamp: string;
  algorithmName?: string;
  isSuccess: boolean;
}

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

/** 样例图片列表响应 */
export interface SampleListResponse {
  code: number;
  data: {
    list: SampleImage[];
    total: number;
  };
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
