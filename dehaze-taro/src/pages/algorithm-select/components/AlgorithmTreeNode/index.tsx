import React from "react";
import { View, Text } from "@tarojs/components";
import { Star, StarOutlined, Info } from "@taroify/icons";
import type { Algorithm } from "dehaze-sdk-js";
import { PUBLISHED_STATUS, getStatusInfo } from "../../utils";

interface AlgorithmTreeNodeProps {
  node: Algorithm;
  level: number;
  expandedKeys: Set<number>;
  favoriteIds: Set<number>;
  onToggleExpand: (id: number) => void;
  onSelect: (algo: Algorithm) => void;
  onToggleFavorite: (algo: Algorithm) => void;
  onShowDetail: (algo: Algorithm) => void;
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
  const isLeaf = !hasChildren;
  const isPublished = node.status === PUBLISHED_STATUS;
  const statusInfo = getStatusInfo(node.status);
  const isFav = favoriteIds.has(node.id);

  return (
    <View key={node.id}>
      <View
        className={`tree-node level-${level} ${isLeaf ? "leaf" : "branch"} ${isPublished ? "selectable" : ""}`}
        onClick={() => {
          if (hasChildren) {
            onToggleExpand(node.id);
          } else {
            onSelect(node);
          }
        }}
      >
        <View className="node-indent" style={{ width: `${level * 16}px` }} />
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
            {isLeaf && (
              <View className={`status-tag ${statusInfo.className}`}>
                <Text>{statusInfo.label}</Text>
              </View>
            )}
          </View>
          {node.description && (
            <Text className="node-desc">{node.description}</Text>
          )}
          {isLeaf && (node.version || node.size || node.flops) && (
            <View className="node-meta">
              {node.version && (
                <Text className="meta-text">v{node.version}</Text>
              )}
              {node.size && <Text className="meta-text">{node.size}</Text>}
              {node.flops && <Text className="meta-text">{node.flops}</Text>}
            </View>
          )}
          {isLeaf && node.type && (
            <View className="node-type">
              <Text className="type-label">{node.type}</Text>
            </View>
          )}
        </View>
        {/* 叶子节点操作按钮 */}
        {isLeaf && (
          <View className="node-actions">
            <View
              className="action-icon"
              onClick={(e) => {
                e.stopPropagation();
                onToggleFavorite(node);
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
              <Info size="16" color="#6b7280" />
            </View>
            {isPublished && (
              <View className="select-btn">
                <Text>使用</Text>
              </View>
            )}
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
