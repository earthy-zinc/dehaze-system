/**
 * 数据集模块类型定义
 *
 * 直接复用 SDK 类型（Dataset / DatasetItemVO / ImageUrlVO / TaskVO），
 * 不再维护独立的字段结构，避免与后端 schema 漂移。
 */
import type {
  Dataset as SDKDataset,
  DatasetItemVO,
  ImageUrlVO,
  DatasetQuery,
  DatasetItemQuery,
  TaskQuery,
} from 'dehaze-sdk-js';

/** 数据集（SDK 类型再导出，便于模块内统一引用） */
export type Dataset = SDKDataset;

/** 数据项 */
export type DatasetItem = DatasetItemVO;

/** 图片文件 */
export type DatasetImage = ImageUrlVO;

/** 数据集查询参数 */
export type { DatasetQuery, DatasetItemQuery, TaskQuery };

/**
 * 数据集树形节点（移动端懒加载展开/收起状态）
 */
export interface DatasetTreeNode extends Dataset {
  /** 子节点（懒加载后填充） */
  children?: DatasetTreeNode[];
  /** 是否已加载过子节点 */
  childrenLoaded?: boolean;
  /** 是否展开 */
  expanded?: boolean;
  /** 缩进层级（根节点为 0） */
  level: number;
}

/**
 * 标注状态筛选（已标注/未标注二分）
 * - annotated: 已标注（hazeLevel 非空）
 * - unannotated: 未标注（hazeLevel 为空）
 */
export type AnnotationFilter = 'annotated' | 'unannotated';

/**
 * 雾霾程度（支持多种规范：light/medium/heavy、beta=0.5 等）
 */
export type HazeLevel = string;

/**
 * 判断图片是否已标注（hazeLevel 非空视为已标注）
 */
export function isImageAnnotated(hazeLevel?: string): boolean {
  return Boolean(hazeLevel);
}
