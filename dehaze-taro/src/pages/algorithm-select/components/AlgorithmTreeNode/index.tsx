import React from "react";
import { View, Text } from "@tarojs/components";
import { Star, StarOutlined } from "@taroify/icons";
import type { AlgorithmSelectNodeVO } from "dehaze-sdk-js";

interface AlgorithmTreeNodeProps {
  node: AlgorithmSelectNodeVO;
  level: number;
  expandedKeys: Set<number>;
  favoriteIds: Set<number>;
  onToggleExpand: (id: number) => void;
  onSelect: (node: AlgorithmSelectNodeVO) => void;
  onToggleFavorite: (id: number) => void;
  onShowDetail: (node: AlgorithmSelectNodeVO) => void;
}

const AlgorithmTreeNode: React.FC<AlgorithmTreeNodeProps> = ({
  node,
  level,
  expandedKeys,
  favoriteIds,
  onToggleExpand,
  onSelect,
  onToggleFavorite,
  onShowDetail,
}) => {
  const hasChildren = node.children && node.children.length > 0;
  const isExpanded = expandedKeys.has(node.id);
  const isLeaf = !hasChildren && node.leaf;
  const isFav = favoriteIds.has(node.id);

  return (
    <View key={node.id}>
      <View
        className={`tree-node level-${level} ${isLeaf ? "leaf" : "branch"} selectable`}
        onClick={() => {
          if (hasChildren) {
            onToggleExpand(node.id);
          } else {
            onSelect(node);
          }
        }}
      >
        <View className="node-indent" style={{ width: `${level * 32}rpx` }} />
        {hasChildren ? (
          <View className="expand-icon">
            <Text>{isExpanded ? "▼" : "▶"}</Text>
          </View>
        ) : (
          <View className="leaf-icon">
            <Text>⚡</Text>
          </View>
        )}
        <View className="node-content">
          <View className="node-header">
            <Text className="node-name">{node.name}</Text>
            {node.type && (
              <Text className="node-type-label">{node.type}</Text>
            )}
          </View>
        </View>
        {isLeaf && (
          <View className="node-actions">
            <View
              className="action-icon"
              onClick={(e) => {
                e.stopPropagation();
                onToggleFavorite(node.id);
              }}
            >
              {isFav ? (
                <Star size="16" color="#f59e0b" />
              ) : (
                <StarOutlined size="16" color="#9ca3af" />
              )}
            </View>
            <View
              className="action-icon"
              onClick={(e) => {
                e.stopPropagation();
                onShowDetail(node);
              }}
            >
              <Text style={{ fontSize: "22rpx", color: "#6b7280" }}>详情</Text>
            </View>
          </View>
        )}
      </View>
      {hasChildren && isExpanded && (
        <View className="tree-children">
          {node.children!.map((child) => (
            <AlgorithmTreeNode
              key={child.id}
              node={child}
              level={level + 1}
              expandedKeys={expandedKeys}
              favoriteIds={favoriteIds}
              onToggleExpand={onToggleExpand}
              onSelect={onSelect}
              onToggleFavorite={onToggleFavorite}
              onShowDetail={onShowDetail}
            />
          ))}
        </View>
      )}
    </View>
  );
};

export default AlgorithmTreeNode;
