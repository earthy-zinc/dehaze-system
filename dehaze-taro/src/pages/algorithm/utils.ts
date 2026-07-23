import type { Algorithm } from "dehaze-sdk-js";

// ==================== 状态定义 ====================

/** 状态信息映射 */
export const STATUS_INFO: Record<
  number,
  {
    label: string;
    color: "default" | "primary" | "success" | "warning" | "danger";
  }
> = {
  0: { label: "草稿", color: "default" },
  1: { label: "测试中", color: "warning" },
  2: { label: "待审核", color: "primary" },
  3: { label: "已发布", color: "success" },
  4: { label: "已停用", color: "default" },
  5: { label: "已归档", color: "default" },
};

/** 状态筛选选项 */
export const STATUS_FILTERS: { label: string; value: number | "" }[] = [
  { label: "全部", value: "" },
  { label: "草稿", value: 0 },
  { label: "测试中", value: 1 },
  { label: "待审核", value: 2 },
  { label: "已发布", value: 3 },
  { label: "已停用", value: 4 },
];

// ==================== 工具函数 ====================

/** 递归展开算法树为平铺列表（含层级缩进信息） */
export interface FlatNode {
  algorithm: Algorithm;
  level: number;
  hasChildren: boolean;
}

export function flattenTree(nodes: Algorithm[], level = 0): FlatNode[] {
  const result: FlatNode[] = [];
  for (const node of nodes) {
    const hasChildren = !!(node.children && node.children.length > 0);
    result.push({ algorithm: node, level, hasChildren });
    if (hasChildren) {
      result.push(...flattenTree(node.children!, level + 1));
    }
  }
  return result;
}

/** 递归过滤算法树（按关键词和状态） */
export function filterTree(
  nodes: Algorithm[],
  keyword: string,
  statusFilter: number | ""
): Algorithm[] {
  const lowerKeyword = keyword.toLowerCase();
  const match = (algo: Algorithm): boolean => {
    const nameMatch =
      !keyword || (algo.name || "").toLowerCase().includes(lowerKeyword);
    const typeMatch =
      !keyword || (algo.type || "").toLowerCase().includes(lowerKeyword);
    const statusMatch = statusFilter === "" || algo.status === statusFilter;
    return (nameMatch || typeMatch) && statusMatch;
  };

  const walk = (list: Algorithm[]): Algorithm[] => {
    const result: Algorithm[] = [];
    for (const node of list) {
      const children = node.children ? walk(node.children) : [];
      if (match(node) || children.length > 0) {
        result.push({
          ...node,
          children: children.length > 0 ? children : undefined,
        });
      }
    }
    return result;
  };
  return walk(nodes);
}

// ==================== 树操作工具函数 ====================

/** 在算法树中更新指定 id 的节点字段 */
export function updateAlgorithmInTree(
  nodes: Algorithm[],
  id: number,
  patch: Partial<Algorithm>
): Algorithm[] {
  return nodes.map((node) => {
    if (node.id === id) return { ...node, ...patch };
    if (node.children)
      return {
        ...node,
        children: updateAlgorithmInTree(node.children, id, patch),
      };
    return node;
  });
}

/** 从算法树中移除指定 id 的节点 */
export function removeAlgorithmFromTree(
  nodes: Algorithm[],
  id: number
): Algorithm[] {
  return nodes
    .filter((node) => node.id !== id)
    .map((node) =>
      node.children
        ? { ...node, children: removeAlgorithmFromTree(node.children, id) }
        : node
    );
}
