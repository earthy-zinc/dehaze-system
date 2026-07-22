/**
 * 算法树形组件
 */

import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
} from 'react-native';
import Icon from '@/components/Icon';
import { theme } from '@/theme';
import type { Algorithm } from '@/types/algorithm';
import AlgorithmCard from './AlgorithmCard';

interface AlgorithmTreeProps {
  tree: Algorithm[];
  favoriteIds: Set<number>;
  compareIds: Set<number>;
  onSelect: (algorithm: Algorithm) => void;
  onToggleFavorite: (algorithm: Algorithm) => void;
  onViewDetail: (algorithm: Algorithm) => void;
  onToggleCompare: (algorithm: Algorithm) => void;
}

/** 判断节点是否为叶子算法（非分类节点） */
function isLeafAlgorithm(node: Algorithm): boolean {
  return !node.children || node.children.length === 0;
}

/** 递归收集所有叶子算法 */
function collectLeafAlgorithms(nodes: Algorithm[]): Algorithm[] {
  const result: Algorithm[] = [];
  for (const node of nodes) {
    if (isLeafAlgorithm(node)) {
      result.push(node);
    } else if (node.children) {
      result.push(...collectLeafAlgorithms(node.children));
    }
  }
  return result;
}

/** 树节点组件 */
const TreeNode: React.FC<{
  node: Algorithm;
  level: number;
  favoriteIds: Set<number>;
  compareIds: Set<number>;
  onSelect: (algorithm: Algorithm) => void;
  onToggleFavorite: (algorithm: Algorithm) => void;
  onViewDetail: (algorithm: Algorithm) => void;
  onToggleCompare: (algorithm: Algorithm) => void;
}> = ({ node, level, favoriteIds, compareIds, onSelect, onToggleFavorite, onViewDetail, onToggleCompare }) => {
  const [expanded, setExpanded] = useState(level < 1);
  const isLeaf = isLeafAlgorithm(node);
  const isCompareSelected = compareIds.has(node.id);

  const handleToggleExpand = useCallback(() => {
    setExpanded(prev => !prev);
  }, []);

  if (isLeaf) {
    return (
      <View style={{ marginLeft: level * 16 }}>
        <AlgorithmCard
          algorithm={node}
          isSelected={isCompareSelected}
          isFavorite={favoriteIds.has(node.id)}
          onSelect={onSelect}
          onToggleFavorite={onToggleFavorite}
          onViewDetail={onViewDetail}
          onToggleCompare={onToggleCompare}
        />
      </View>
    );
  }

  return (
    <View style={{ marginLeft: level * 16 }}>
      {/* 分类节点 */}
      <TouchableOpacity
        style={styles.categoryNode}
        onPress={handleToggleExpand}
        activeOpacity={0.7}
      >
        <Icon
          name={expanded ? 'chevron-down' : 'chevron-forward'}
          size={18}
          color={theme.colors.text.secondary}
        />
        <Icon
          name={expanded ? 'folder-open' : 'folder'}
          size={18}
          color={theme.colors.primary}
        />
        <Text style={styles.categoryName}>{node.name}</Text>
        {node.children && (
          <View style={styles.countBadge}>
            <Text style={styles.countText}>
              {collectLeafAlgorithms(node.children).length}
            </Text>
          </View>
        )}
      </TouchableOpacity>

      {/* 子节点 */}
      {expanded && node.children && (
        <View>
          {node.children.map(child => (
            <TreeNode
              key={child.id}
              node={child}
              level={level + 1}
              favoriteIds={favoriteIds}
              compareIds={compareIds}
              onSelect={onSelect}
              onToggleFavorite={onToggleFavorite}
              onViewDetail={onViewDetail}
              onToggleCompare={onToggleCompare}
            />
          ))}
        </View>
      )}
    </View>
  );
};

const AlgorithmTree: React.FC<AlgorithmTreeProps> = ({
  tree,
  favoriteIds,
  compareIds,
  onSelect,
  onToggleFavorite,
  onViewDetail,
  onToggleCompare,
}) => {
  if (tree.length === 0) {
    return (
      <View style={styles.emptyContainer}>
        <Icon name="folder-open-outline" size={48} color={theme.colors.text.tertiary} />
        <Text style={styles.emptyText}>暂无可用算法</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {tree.map(node => (
        <TreeNode
          key={node.id}
          node={node}
          level={0}
          favoriteIds={favoriteIds}
          compareIds={compareIds}
          onSelect={onSelect}
          onToggleFavorite={onToggleFavorite}
          onViewDetail={onViewDetail}
          onToggleCompare={onToggleCompare}
        />
      ))}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  categoryNode: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: theme.spacing.sm,
    paddingHorizontal: theme.spacing.sm,
    gap: theme.spacing.xs,
    marginBottom: theme.spacing.xs,
  },
  categoryName: {
    flex: 1,
    fontSize: theme.typography.sizes.body,
    fontWeight: theme.typography.weights.semibold,
    color: theme.colors.text.primary,
  },
  countBadge: {
    backgroundColor: theme.colors.background.tertiary,
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 10,
  },
  countText: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.secondary,
    fontWeight: theme.typography.weights.medium,
  },
  emptyContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: theme.spacing.xxxl,
    gap: theme.spacing.md,
  },
  emptyText: {
    fontSize: theme.typography.sizes.body,
    color: theme.colors.text.tertiary,
  },
});

export default AlgorithmTree;
