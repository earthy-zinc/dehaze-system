import type { Algorithm } from "dehaze-sdk-js";

// 算法状态：4=已发布
export const PUBLISHED_STATUS = 4;
export const FAVORITE_STORAGE_KEY = "favorite_algorithms";

// 算法类型推荐权重（数值越高越优先推荐）
export const TYPE_WEIGHT: Record<string, number> = {
  // 深度学习类
  cnn: 10,
  gan: 9,
  transformer: 10,
  深度学习: 10,
  deeplab: 9,
  // 混合类
  混合: 7,
  hybrid: 7,
  // 传统类
  传统: 5,
  dcp: 5,
  retinex: 5,
  暗通道: 5,
};

// 获取算法类型权重
export function getTypeWeight(type: string): number {
  const lower = (type || "").toLowerCase();
  for (const key of Object.keys(TYPE_WEIGHT)) {
    if (lower.includes(key.toLowerCase())) return TYPE_WEIGHT[key];
  }
  return 6;
}

// 递归收集所有叶子算法
export function collectLeafAlgorithms(nodes: Algorithm[]): Algorithm[] {
  const result: Algorithm[] = [];
  const walk = (list: Algorithm[]) => {
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

// 状态信息
export function getStatusInfo(status?: number) {
  switch (status) {
    case 1:
      return { label: "草稿", className: "status-draft" };
    case 2:
      return { label: "测试中", className: "status-testing" };
    case 3:
      return { label: "待审核", className: "status-pending" };
    case 4:
      return { label: "已发布", className: "status-published" };
    case 5:
      return { label: "已停用", className: "status-disabled" };
    case 6:
      return { label: "已归档", className: "status-archived" };
    default:
      return { label: "未知", className: "status-unknown" };
  }
}

// 递归搜索过滤
export function filterTree(nodes: Algorithm[], keyword: string): Algorithm[] {
  if (!keyword) return nodes;
  const lower = keyword.toLowerCase();
  const result: Algorithm[] = [];
  for (const node of nodes) {
    const nameMatch = node.name?.toLowerCase().includes(lower);
    const descMatch = node.description?.toLowerCase().includes(lower);
    if (node.children && node.children.length > 0) {
      const filteredChildren = filterTree(node.children, keyword);
      if (filteredChildren.length > 0 || nameMatch) {
        result.push({ ...node, children: filteredChildren });
      }
    } else if (nameMatch || descMatch) {
      result.push(node);
    }
  }
  return result;
}
