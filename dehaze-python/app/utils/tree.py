"""
树形结构工具函数

提供树路径生成、子节点收集等通用纯函数。
所有函数无 DB / ORM 依赖，可在任意层使用。
"""

from collections import deque


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
