import React from "react";
import { View, ScrollView } from "@tarojs/components";
import EmptyState from "@/components/common/EmptyState";
import ErrorState from "@/components/common/ErrorState";
import type { Dataset } from "../../services/types";
import DatasetCard from "../DatasetCard";
import "./DatasetList.less";

interface DatasetListProps {
  datasets: Dataset[];
  loading?: boolean;
  error?: string | null;
  onRetry?: () => void;
  onLoadMore?: () => void;
  hasMore?: boolean;
  onDatasetClick?: (dataset: Dataset) => void;
  // 树形结构相关
  expandedIds: number[];
  childrenMap: Record<number, Dataset[]>;
  childrenLoading: Record<number, boolean>;
  onToggleExpand?: (id: number) => void;
  // CRUD 相关
  onAddChild?: (parent: Dataset) => void;
  onEdit?: (dataset: Dataset) => void;
  onDelete?: (dataset: Dataset) => void;
  className?: string;
  browseMode?: boolean;
}

// 递归树节点渲染
const TreeNode: React.FC<{
  dataset: Dataset;
  depth: number;
  expandedIds: number[];
  childrenMap: Record<number, Dataset[]>;
  childrenLoading: Record<number, boolean>;
  onDatasetClick?: (dataset: Dataset) => void;
  onToggleExpand?: (id: number) => void;
  onAddChild?: (parent: Dataset) => void;
  onEdit?: (dataset: Dataset) => void;
  onDelete?: (dataset: Dataset) => void;
}> = ({
  dataset,
  depth,
  expandedIds,
  childrenMap,
  childrenLoading,
  onDatasetClick,
  onToggleExpand,
  onAddChild,
  onEdit,
  onDelete,
}) => {
  const isExpanded = expandedIds.includes(dataset.id);
  const hasChildren = dataset.hasChildren === true;
  const children = childrenMap[dataset.id];
  const isLoadingChildren = childrenLoading[dataset.id];

  return (
    <View className="tree-node">
      <DatasetCard
        dataset={dataset}
        depth={depth}
        expanded={isExpanded}
        hasChildren={hasChildren}
        loading={isLoadingChildren}
        onClick={() => onDatasetClick?.(dataset)}
        onToggleExpand={() => onToggleExpand?.(dataset.id)}
        onAddChild={onAddChild ? () => onAddChild(dataset) : undefined}
        onEdit={onEdit ? () => onEdit(dataset) : undefined}
        onDelete={onDelete ? () => onDelete(dataset) : undefined}
      />
      {/* 递归渲染子节点 */}
      {isExpanded && children && children.length > 0 && (
        <View className="tree-children">
          {children.map((child) => (
            <TreeNode
              key={child.id}
              dataset={child}
              depth={depth + 1}
              expandedIds={expandedIds}
              childrenMap={childrenMap}
              childrenLoading={childrenLoading}
              onDatasetClick={onDatasetClick}
              onToggleExpand={onToggleExpand}
              onAddChild={onAddChild}
              onEdit={onEdit}
              onDelete={onDelete}
            />
          ))}
        </View>
      )}
      {/* 子节点加载中 */}
      {isExpanded &&
        isLoadingChildren &&
        (!children || children.length === 0) && (
          <View className="children-loading">
            <View className="loading-spinner" />
          </View>
        )}
      {/* 展开但无子节点 */}
      {isExpanded &&
        !isLoadingChildren &&
        children &&
        children.length === 0 && (
          <View className="empty-children">
            <View className="empty-children-text">暂无子数据集</View>
          </View>
        )}
    </View>
  );
};

const DatasetList: React.FC<DatasetListProps> = ({
  datasets,
  loading = false,
  error,
  onRetry,
  onLoadMore,
  hasMore = false,
  onDatasetClick,
  expandedIds,
  childrenMap,
  childrenLoading,
  onToggleExpand,
  onAddChild,
  onEdit,
  onDelete,
  className = "",
}) => {
  const handleScrollToLower = () => {
    if (onLoadMore && hasMore && !loading) {
      onLoadMore();
    }
  };

  if (error && datasets.length === 0 && !loading) {
    return (
      <View className={`dataset-list ${className}`}>
        <ErrorState message={error} onRetry={onRetry} />
      </View>
    );
  }

  if (datasets.length === 0 && !loading) {
    return (
      <View className={`dataset-list ${className}`}>
        <EmptyState type="dataset" />
      </View>
    );
  }

  return (
    <ScrollView
      className={`dataset-list ${className}`}
      scrollY
      onScrollToLower={handleScrollToLower}
      lowerThreshold={100}
    >
      <View className="list-content">
        {datasets.map((dataset) => (
          <TreeNode
            key={dataset.id}
            dataset={dataset}
            depth={0}
            expandedIds={expandedIds}
            childrenMap={childrenMap}
            childrenLoading={childrenLoading}
            onDatasetClick={onDatasetClick}
            onToggleExpand={onToggleExpand}
            onAddChild={onAddChild}
            onEdit={onEdit}
            onDelete={onDelete}
          />
        ))}

        {/* 加载更多触发器 */}
        {hasMore && (
          <View className="load-more-trigger">
            <View className="loading-spinner" />
            <View className="loading-text">加载中...</View>
          </View>
        )}

        {/* 没有更多数据 */}
        {!hasMore && datasets.length > 0 && (
          <View className="no-more">
            <View className="no-more-text">已加载全部数据</View>
          </View>
        )}
      </View>
    </ScrollView>
  );
};

export default DatasetList;
