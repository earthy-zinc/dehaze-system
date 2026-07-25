import React, { useEffect, useState } from "react";
import { View, ScrollView } from "@tarojs/components";
import { Arrow, Add } from "@taroify/icons";
import { confirmDialog } from "@/utils/dialog";

// 组件导入
import SearchBar from "@/components/common/SearchBar";
import FilterTabs from "@/components/common/FilterTabs";
import EmptyState from "@/components/common/EmptyState";

// 本地组件导入
import DatasetList from "./components/DatasetList";
import DatasetInfo from "./components/DatasetInfo";
import ImageGrid from "./components/ImageGrid";
import DatasetFormDialog, {
  DatasetFormData,
} from "./components/DatasetFormDialog";

// Store 和类型
import { DatasetProvider, useDataset } from "./store/datasetStore";
import type { Dataset } from "./services/types";
import {
  AnnotationFilter,
  ANNOTATION_FILTER_LABELS,
} from "./services/imageUtils";

import "./index.less";

// 弹窗状态
interface DialogState {
  visible: boolean;
  mode: "create" | "edit";
  dataset: Dataset | null;
  defaultParentId: number;
}

// 主组件内容
const DatasetContent: React.FC = () => {
  const {
    state,
    setView,
    setCurrentDatasetId,
    fetchDatasets,
    fetchDatasetDetail,
    fetchImages,
    setSearchKeyword,
    setImageSearchKeyword,
    setAnnotationFilter,
    resetImages,
    toggleExpand,
    fetchDatasetOptions,
    createDataset,
    updateDataset,
    deleteDataset,
  } = useDataset();

  // 本地状态
  const [searchInputValue, setSearchInputValue] = useState("");
  const [dialog, setDialog] = useState<DialogState>({
    visible: false,
    mode: "create",
    dataset: null,
    defaultParentId: 0,
  });

  // 初始化加载数据集列表和下拉选项
  useEffect(() => {
    fetchDatasets(1, "", false);
    fetchDatasetOptions();
  }, [fetchDatasets, fetchDatasetOptions]);

  // 搜索处理
  const handleSearch = (keyword: string) => {
    setSearchKeyword(keyword);
    setSearchInputValue(keyword);
    if (state.currentView === "list") {
      fetchDatasets(1, keyword, false);
    } else {
      setImageSearchKeyword(keyword);
      fetchImages(
        state.currentDatasetId!,
        1,
        state.currentAnnotationFilter,
        keyword,
        false
      );
    }
  };

  // 清除搜索
  const handleClearSearch = () => {
    handleSearch("");
  };

  // 数据集点击处理
  const handleDatasetClick = (dataset: Dataset) => {
    setCurrentDatasetId(dataset.id);
    setView("detail");
    resetImages();

    // 加载数据集详情
    fetchDatasetDetail(dataset.id);

    // 加载图片列表（默认显示已标注）
    fetchImages(dataset.id, 1, "annotated", "", false);
  };

  // 返回列表处理
  const handleBackToList = () => {
    setView("list");
    setCurrentDatasetId(null);
    setSearchInputValue("");
    setImageSearchKeyword("");
  };

  // 标注状态筛选处理（已标注/未标注二分）
  const handleAnnotationFilterChange = (filter: string) => {
    const annotationFilter = filter as AnnotationFilter;
    setAnnotationFilter(annotationFilter);
    fetchImages(
      state.currentDatasetId!,
      1,
      annotationFilter,
      state.imageSearchKeyword,
      false
    );
  };

  // 加载更多数据集
  const handleLoadMoreDatasets = () => {
    if (state.datasetsHasMore && !state.datasetsLoading) {
      fetchDatasets(state.datasetsPage + 1, state.searchKeyword, true);
    }
  };

  // 加载更多图片
  const handleLoadMoreImages = () => {
    if (state.imagesHasMore && !state.imagesLoading) {
      fetchImages(
        state.currentDatasetId!,
        state.imagesPage + 1,
        state.currentAnnotationFilter,
        state.imageSearchKeyword,
        true
      );
    }
  };

  // 新增根数据集
  const handleAddRoot = () => {
    setDialog({
      visible: true,
      mode: "create",
      dataset: null,
      defaultParentId: 0,
    });
  };

  // 新增子数据集
  const handleAddChild = (parent: Dataset) => {
    setDialog({
      visible: true,
      mode: "create",
      dataset: null,
      defaultParentId: parent.id,
    });
  };

  // 编辑数据集
  const handleEdit = (dataset: Dataset) => {
    setDialog({
      visible: true,
      mode: "edit",
      dataset,
      defaultParentId: dataset.parentId ?? 0,
    });
  };

  // 删除数据集（带确认）
  const handleDelete = async (dataset: Dataset) => {
    const confirmed = await confirmDialog({
      title: "确认删除",
      content: `确定要删除数据集「${dataset.name}」吗？此操作不可恢复。`,
      confirmText: "删除",
      confirmColor: "#ef4444",
      cancelText: "取消",
    });
    if (!confirmed) return;
    await deleteDataset(dataset.id);
  };

  // 表单提交处理
  const handleSubmit = async (data: DatasetFormData): Promise<boolean> => {
    if (dialog.mode === "create") {
      return await createDataset({
        parentId: data.parentId,
        type: data.type,
        name: data.name,
        description: data.description,
        status: data.status,
      });
    }
    if (!dialog.dataset) return false;
    return await updateDataset(dialog.dataset.id, {
      type: data.type,
      name: data.name,
      description: data.description,
      status: data.status,
    });
  };

  // 关闭弹窗
  const handleCloseDialog = () => {
    setDialog((prev) => ({ ...prev, visible: false }));
  };

  // 标注状态筛选标签配置（已标注/未标注二分）
  const stats = state.currentDataset?.statistics;
  const annotatedCount = stats?.annotatedCount ?? 0;
  const unannotatedCount = stats?.unannotatedCount ?? 0;
  const annotationFilterTabs = state.currentDataset
    ? [
        {
          key: "annotated",
          label: ANNOTATION_FILTER_LABELS.annotated,
          count: annotatedCount,
        },
        {
          key: "unannotated",
          label: ANNOTATION_FILTER_LABELS.unannotated,
          count: unannotatedCount,
        },
      ]
    : [];

  return (
    <View className="dataset-page">
      {/* 搜索栏 */}
      <View className="search-section">
        <SearchBar
          placeholder={
            state.currentView === "detail" ? "搜索图片..." : "搜索数据集..."
          }
          value={searchInputValue}
          onSearch={handleSearch}
          onClear={handleClearSearch}
        />
      </View>

      {/* 列表视图 */}
      {state.currentView === "list" && (
        <View className="list-view">
          {/* 顶部操作栏 */}
          <View className="action-bar">
            <View className="add-btn" onClick={handleAddRoot}>
              <Add size="16" color="#ffffff" />
              <View className="add-btn-text">新增数据集</View>
            </View>
          </View>

          <DatasetList
            datasets={state.datasets}
            loading={state.datasetsLoading}
            error={state.datasetsError}
            onRetry={() => fetchDatasets(1, state.searchKeyword, false)}
            hasMore={state.datasetsHasMore}
            onLoadMore={handleLoadMoreDatasets}
            onDatasetClick={handleDatasetClick}
            expandedIds={state.expandedIds}
            childrenMap={state.childrenMap}
            childrenLoading={state.childrenLoading}
            onToggleExpand={toggleExpand}
            onAddChild={handleAddChild}
            onEdit={handleEdit}
            onDelete={handleDelete}
          />
        </View>
      )}

      {/* 详情视图 */}
      {state.currentView === "detail" && state.currentDataset && (
        <View className="detail-view">
          {/* 返回按钮 */}
          <View className="back-section">
            <View className="back-btn" onClick={handleBackToList}>
              <Arrow />
              <View className="back-text">返回列表</View>
            </View>
          </View>

          <ScrollView className="detail-content" scrollY>
            {/* 数据集信息 */}
            <DatasetInfo dataset={state.currentDataset} />

            {/* 标注状态筛选 */}
            <View className="filter-section">
              <FilterTabs
                tabs={annotationFilterTabs}
                activeKey={state.currentAnnotationFilter}
                onChange={handleAnnotationFilterChange}
              />
            </View>

            {/* 图片网格 */}
            <View className="images-section">
              {state.images.length === 0 && !state.imagesLoading ? (
                <EmptyState
                  type="image"
                  title="暂无图片"
                  description="该数据集中没有符合条件的图片"
                />
              ) : (
                <ImageGrid images={state.images} />
              )}
            </View>

            {/* 加载更多触发器 */}
            {state.imagesHasMore && (
              <View className="load-more-section">
                <View
                  className="load-more-trigger"
                  onClick={handleLoadMoreImages}
                >
                  {state.imagesLoading ? (
                    <View className="loading-spinner" />
                  ) : (
                    <View className="load-more-text">加载更多</View>
                  )}
                </View>
              </View>
            )}
          </ScrollView>
        </View>
      )}

      {/* 新增/编辑弹窗 */}
      <DatasetFormDialog
        visible={dialog.visible}
        mode={dialog.mode}
        dataset={dialog.dataset}
        options={state.datasetOptions}
        defaultParentId={dialog.defaultParentId}
        onSubmit={handleSubmit}
        onClose={handleCloseDialog}
      />
    </View>
  );
};

// 包装组件
const DatasetPage: React.FC = () => {
  return (
    <DatasetProvider>
      <DatasetContent />
    </DatasetProvider>
  );
};

export default DatasetPage;
