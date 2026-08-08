/**
 * 数据集浏览版（L2）
 * 公开/共享浏览：数据集列表、详情查看、图片浏览
 * 无创建/编辑/删除管理操作
 */
import React, { useEffect, useState } from "react";
import { View, ScrollView, Text as TaroText } from "@tarojs/components";
import Taro from "@tarojs/taro";
import { Arrow } from "@taroify/icons";

import SearchBar from "@/components/common/SearchBar";
import FilterTabs from "@/components/common/FilterTabs";
import EmptyState from "@/components/common/EmptyState";

import DatasetList from "./components/DatasetList";
import DatasetInfo from "./components/DatasetInfo";
import ImageGrid from "./components/ImageGrid";

import { DatasetProvider, useDataset } from "./store/datasetStore";
import type { Dataset } from "./services/types";
import {
  AnnotationFilter,
  ANNOTATION_FILTER_LABELS,
} from "./services/imageUtils";

import "./index.less";

const DatasetBrowseContent: React.FC = () => {
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
  } = useDataset();

  const [searchInputValue, setSearchInputValue] = useState("");
  const [routeError, setRouteError] = useState<string | null>(null);

  useEffect(() => {
    const params = Taro.getCurrentInstance()?.router?.params;
    const datasetIdStr = params?.datasetId || params?.id;
    if (datasetIdStr) {
      const id = Number(datasetIdStr);
      if (!Number.isNaN(id)) {
        setCurrentDatasetId(id);
        setView("detail");
        fetchDatasetDetail(id);
        fetchImages(id, 1, "annotated", "", false);
      } else {
        setRouteError("无效的数据集ID");
      }
    }
    fetchDatasets(1, "", false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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

  const handleClearSearch = () => {
    handleSearch("");
  };

  const handleDatasetClick = (dataset: Dataset) => {
    setCurrentDatasetId(dataset.id);
    setView("detail");
    resetImages();
    fetchDatasetDetail(dataset.id);
    fetchImages(dataset.id, 1, "annotated", "", false);
  };

  const handleBackToList = () => {
    setView("list");
    setCurrentDatasetId(null);
    setSearchInputValue("");
    setImageSearchKeyword("");
  };

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

  const handleLoadMoreDatasets = () => {
    if (state.datasetsHasMore && !state.datasetsLoading) {
      fetchDatasets(state.datasetsPage + 1, state.searchKeyword, true);
    }
  };

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

      {routeError && (
        <View className="error-view">
          <TaroText className="error-text">{routeError}</TaroText>
          <View className="back-btn" onClick={() => Taro.navigateBack()}>
            <TaroText>返回</TaroText>
          </View>
        </View>
      )}

      {state.currentView === "list" && !routeError && (
        <View className="list-view">
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
            onAddChild={() => {}}
            onEdit={() => {}}
            onDelete={() => {}}
            browseMode
          />
        </View>
      )}

      {state.currentView === "detail" && state.currentDataset && (
        <View className="detail-view">
          <View className="back-section">
            <View className="back-btn" onClick={handleBackToList}>
              <Arrow />
              <View className="back-text">返回列表</View>
            </View>
          </View>

          <ScrollView className="detail-content" scrollY>
            <DatasetInfo dataset={state.currentDataset} />

            <View className="filter-section">
              <FilterTabs
                tabs={annotationFilterTabs}
                activeKey={state.currentAnnotationFilter}
                onChange={handleAnnotationFilterChange}
              />
            </View>

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
    </View>
  );
};

const DatasetBrowsePage: React.FC = () => {
  return (
    <DatasetProvider>
      <DatasetBrowseContent />
    </DatasetProvider>
  );
};

export default DatasetBrowsePage;
