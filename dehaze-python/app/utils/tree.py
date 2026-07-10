"""
树形结构工具函数

提供树构建、树路径生成、子节点收集等通用纯函数。
所有函数无 DB / ORM 依赖，可在任意层使用。
"""

from collections import deque
from typing import Any


def generate_tree_path(parent_tree_path: str | None, parent_id: int) -> str:
    """
    生成树路径

    Args:
        parent_tree_path: 父节点的树路径
        parent_id: 父节点ID

    Returns:
        新节点的树路径

    Usage:
        # 父节点 tree_path 为 "0,1,2"，parent_id 为 2
        # 返回 "0,1,2,2"
        tree_path = generate_tree_path(parent.tree_path, parent.id)
    """
    if parent_id == 0:
        return "0"

    if not parent_tree_path:
        return str(parent_id)

    return f"{parent_tree_path},{parent_id}"


def bfs_collect_ids(
    start_id: int,
    children_map: dict[int, list],
    *,
    include_start: bool = True,
) -> list[int]:
    """
    BFS 遍历收集节点 ID

    Args:
        start_id: 起始节点ID
        children_map: 父子关系映射 {parent_id: [child1, child2, ...]}
        include_start: 是否包含起始节点

    Returns:
        节点ID列表
    """
    queue = deque([start_id])
    collected = [start_id] if include_start else []

    while queue:
        current_id = queue.popleft()
        for child in children_map.get(current_id, []):
            collected.append(child.id if hasattr(child, 'id') else child)
            queue.append(child.id if hasattr(child, 'id') else child)

    return collected


def build_tree_from_list(
    items: list,
    id_field: str = "id",
    parent_id_field: str = "parent_id",
    root_parent_id: int = 0,
) -> list[dict]:
    """
    从扁平列表构建树形结构

    Args:
        items: 扁平列表（支持对象或字典）
        id_field: ID字段名
        parent_id_field: 父ID字段名
        root_parent_id: 根节点的父ID值

    Returns:
        树形结构列表
    """
    if not items:
        return []

    # 构建映射
    def get_id(item):
        return item[id_field] if isinstance(item, dict) else getattr(item, id_field)

    def get_parent_id(item):
        return item[parent_id_field] if isinstance(item, dict) else getattr(item, parent_id_field)

    item_map = {get_id(item): {"item": item, "children": []} for item in items}

    # 构建树
    roots = []
    for item in items:
        item_id = get_id(item)
        parent_id = get_parent_id(item)

        if parent_id == root_parent_id:
            roots.append(item_map[item_id])
        elif parent_id in item_map:
            item_map[parent_id]["children"].append(item_map[item_id])

    return roots


def build_tree(
    items: list[dict[str, Any]],
    *,
    id_key: str = "id",
    parent_key: str = "parentId",
    children_key: str = "children",
    root_value: int = 0,
) -> list[dict[str, Any]]:
    """将扁平字典列表构建为树形结构（O(n) 复杂度）"""
    node_map = {}
    for item in items:
        item.setdefault(children_key, [])
        node_map[item[id_key]] = item

    roots = []
    for item in items:
        parent_id = item[parent_key]
        if parent_id == root_value or parent_id not in node_map:
            roots.append(item)
        else:
            node_map[parent_id][children_key].append(item)
    return roots


def build_tree_options(
    items: list,
    *,
    id_attr: str = "id",
    label_attr: str = "name",
    parent_attr: str = "parent_id",
    root_value: int = 0,
) -> list[dict[str, Any]]:
    """从 ORM 实体列表构建下拉选项树（value/label/children 结构）"""
    node_map = {}
    for item in items:
        node = {
            "value": getattr(item, id_attr),
            "label": getattr(item, label_attr),
            "children": [],
        }
        node_map[getattr(item, id_attr)] = node

    roots = []
    for item in items:
        pid = getattr(item, parent_attr)
        node = node_map[getattr(item, id_attr)]
        if pid == root_value or pid not in node_map:
            roots.append(node)
        else:
            node_map[pid]["children"].append(node)
    return roots
