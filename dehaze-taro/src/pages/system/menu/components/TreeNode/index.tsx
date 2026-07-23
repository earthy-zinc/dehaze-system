import React from "react";
import { View, Text } from "@tarojs/components";
import { SwipeCell, Button, Cell, Tag } from "@taroify/core";
import { Arrow, Add, Edit, Delete } from "@taroify/icons";
import type { MenuVO } from "dehaze-sdk-js";
import { MENU_TYPE_CONFIG } from "../../constants";

interface TreeNodeProps {
  node: MenuVO;
  depth: number;
  expandedKeys: number[];
  onToggle: (id: number) => void;
  onAddChild: (parentId: number) => void;
  onEdit: (node: MenuVO) => void;
  onDelete: (node: MenuVO) => void;
  hasPermission: (perm: string) => boolean;
}

const TreeNode: React.FC<TreeNodeProps> = ({
  node,
  depth,
  expandedKeys,
  onToggle,
  onAddChild,
  onEdit,
  onDelete,
  hasPermission,
}) => {
  const hasChildren = node.children && node.children.length > 0;
  const isExpanded = expandedKeys.includes(node.id!);
  const typeConfig = node.type ? MENU_TYPE_CONFIG[node.type] : null;

  return (
    <View className="menu-tree-node">
      <SwipeCell className="menu-swipe-cell">
        <SwipeCell.Actions side="right">
          {hasPermission("sys:menu:add") && (
            <Button
              className="action-btn add-btn"
              size="small"
              onClick={() => onAddChild(node.id!)}
            >
              <Add />
              子级
            </Button>
          )}
          {hasPermission("sys:menu:edit") && (
            <Button
              className="action-btn edit-btn"
              size="small"
              onClick={() => onEdit(node)}
            >
              <Edit />
              编辑
            </Button>
          )}
          {hasPermission("sys:menu:delete") && (
            <Button
              className="action-btn delete-btn"
              size="small"
              onClick={() => onDelete(node)}
            >
              <Delete />
              删除
            </Button>
          )}
        </SwipeCell.Actions>
        <Cell
          className="menu-cell"
          style={{ paddingLeft: `${16 + depth * 20}px` }}
          onClick={() => hasChildren && onToggle(node.id!)}
        >
          <View className="menu-row">
            {hasChildren ? (
              <View className="menu-toggle">
                <Arrow
                  className={isExpanded ? "arrow-expanded" : "arrow-collapsed"}
                />
              </View>
            ) : (
              <View className="menu-toggle-placeholder" />
            )}
            <View className="menu-info">
              <View className="menu-name-row">
                <Text className="menu-name">{node.name}</Text>
                {typeConfig && (
                  <Tag color={typeConfig.color} size="small">
                    {typeConfig.label}
                  </Tag>
                )}
                {node.visible === 0 && (
                  <Tag color="default" size="small">
                    隐藏
                  </Tag>
                )}
              </View>
              <View className="menu-meta">
                {node.routePath && (
                  <Text className="meta-text">路由: {node.routePath}</Text>
                )}
                {node.perm && (
                  <Text className="meta-text">权限: {node.perm}</Text>
                )}
                <Text className="meta-text">排序: {node.sort ?? 0}</Text>
              </View>
            </View>
          </View>
        </Cell>
      </SwipeCell>
      {hasChildren && isExpanded && (
        <View className="menu-children">
          {node.children!.map((child) => (
            <TreeNode
              key={child.id}
              node={child}
              depth={depth + 1}
              expandedKeys={expandedKeys}
              onToggle={onToggle}
              onAddChild={onAddChild}
              onEdit={onEdit}
              onDelete={onDelete}
              hasPermission={hasPermission}
            />
          ))}
        </View>
      )}
    </View>
  );
};

export default TreeNode;
