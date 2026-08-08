import type { AlgorithmSelectNodeVO } from "dehaze-sdk-js";

/** 算法选择树节点（兼容 AlgorithmSelectNodeVO） */
export type TreeNode = AlgorithmSelectNodeVO;

/** 算法类型推荐权重（数值越高越优先推荐） */
export const TYPE_WEIGHT: Record<string, number> = {
  cnn: 10,
  gan: 9,
  transformer: 10,
  深度学习: 10,
  deeplab: 9,
  混合: 7,
  hybrid: 7,
  传统: 5,
  dcp: 5,
  retinex: 5,
  暗通道: 5,
};

/** 获取算法类型权重 */
export function getTypeWeight(type: string): number {
  const lower = (type || "").toLowerCase();
  for (const key of Object.keys(TYPE_WEIGHT)) {
    if (lower.includes(key.toLowerCase())) return TYPE_WEIGHT[key];
  }
  return 6;
}

/** 递归收集所有叶子算法节点 */
export function collectLeafAlgorithms(nodes: TreeNode[]): TreeNode[] {
  const result: TreeNode[] = [];
  const walk = (list: TreeNode[]) => {
    for (const node of list) {
      if (node.children && node.children.length > 0) {
        walk(node.children);
      } else {
        result.push(node);
      }
    }
  };
  walk(nodes);
  return result;
}

/** 搜索历史最多保存条数 */
export const SEARCH_HISTORY_MAX = 10;
export const SEARCH_HISTORY_KEY = "alg_select_search_history";

/** 读取搜索历史 */
export function getSearchHistory(): string[] {
  try {
    const raw = localStorage.getItem(SEARCH_HISTORY_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

/** 保存搜索历史（去重 + 上限截断） */
export function saveSearchHistory(keyword: string): string[] {
  const list = getSearchHistory().filter((k) => k !== keyword);
  list.unshift(keyword);
  const trimmed = list.slice(0, SEARCH_HISTORY_MAX);
  localStorage.setItem(SEARCH_HISTORY_KEY, JSON.stringify(trimmed));
  return trimmed;
}

/** 清空搜索历史 */
export function clearSearchHistory() {
  localStorage.removeItem(SEARCH_HISTORY_KEY);
}

/** 筛选条件 */
export interface FilterConditions {
  types: string[];
  speed: string;
  quality: string;
  scenes: string[];
  minRating: number;
}

export const DEFAULT_FILTER: FilterConditions = {
  types: [],
  speed: "",
  quality: "",
  scenes: [],
  minRating: 0,
};

/** 对比列表上限 */
export const COMPARE_MAX = 3;
