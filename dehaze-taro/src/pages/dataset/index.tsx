import React, { useEffect, useState } from 'react'
import { View, ScrollView } from '@tarojs/components'
import { Arrow } from '@taroify/icons'
import Taro from '@tarojs/taro'

// 组件导入
import SearchBar from '@/components/common/SearchBar'
import FilterTabs from '@/components/common/FilterTabs'
import EmptyState from '@/components/common/EmptyState'

// 本地组件导入
import DatasetList from './components/DatasetList'
import DatasetInfo from './components/DatasetInfo'
import ImageGrid from './components/ImageGrid'

// Store 和类型
import { DatasetProvider, useDataset } from './store/datasetStore'
import { Dataset, DatasetImage } from './services/types'

import './index.less'

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
    setImageType,
    resetImages,
  } = useDataset()

  // 本地状态
  const [searchInputValue, setSearchInputValue] = useState('')

  // 初始化加载数据集列表
  useEffect(() => {
    fetchDatasets(1, '', false)
  }, [])

  // 搜索处理
  const handleSearch = (keyword: string) => {
    setSearchKeyword(keyword)
    setSearchInputValue(keyword)
    if (state.currentView === 'list') {
      fetchDatasets(1, keyword, false)
    } else {
      setImageSearchKeyword(keyword)
      fetchImages(state.currentDatasetId!, 1, state.currentImageType, keyword, false)
    }
  }

  // 清除搜索
  const handleClearSearch = () => {
    handleSearch('')
  }

  // 数据集点击处理
  const handleDatasetClick = (dataset: Dataset) => {
    setCurrentDatasetId(dataset.id)
    setView('detail')
    resetImages()

    // 加载数据集详情
    fetchDatasetDetail(dataset.id)

    // 加载图片列表
    fetchImages(dataset.id, 1, 'all', '', false)
  }

  // 返回列表处理
  const handleBackToList = () => {
    setView('list')
    setCurrentDatasetId(null)
    setSearchInputValue('')
    setImageSearchKeyword('')
  }

  // 图片类型筛选处理
  const handleImageTypeFilter = (type: string) => {
    setImageType(type as 'all' | 'foggy' | 'clear' | 'annotated')
    fetchImages(state.currentDatasetId!, 1, type as any, state.imageSearchKeyword, false)
  }

  // 加载更多数据集
  const handleLoadMoreDatasets = () => {
    if (state.datasetsHasMore && !state.datasetsLoading) {
      fetchDatasets(state.datasetsPage + 1, state.searchKeyword, true)
    }
  }

  // 加载更多图片
  const handleLoadMoreImages = () => {
    if (state.imagesHasMore && !state.imagesLoading) {
      fetchImages(
        state.currentDatasetId!,
        state.imagesPage + 1,
        state.currentImageType,
        state.imageSearchKeyword,
        true
      )
    }
  }

  // 图片点击处理
  const handleImageClick = (image: DatasetImage) => {
    console.log('Image clicked:', image.filename)
  }

  // 图片类型筛选标签配置
  const imageTypeTabs = state.currentDataset ? [
    { key: 'all', label: '全部', count: state.currentDataset.total_images },
    { key: 'foggy', label: '有雾', count: state.currentDataset.foggy_count },
    { key: 'clear', label: '无雾', count: state.currentDataset.clear_count },
    { key: 'annotated', label: '标注', count: state.currentDataset.annotated_count },
  ] : []

  return (
    <View className="dataset-page">
      {/* 搜索栏 */}
      <View className="search-section">
        <SearchBar
          placeholder={state.currentView === 'detail' ? '搜索图片...' : '搜索数据集...'}
          value={searchInputValue}
          onSearch={handleSearch}
          onClear={handleClearSearch}
        />
      </View>

      {/* 列表视图 */}
      {state.currentView === 'list' && (
        <View className="list-view">
          <DatasetList
            datasets={state.datasets}
            loading={state.datasetsLoading}
            hasMore={state.datasetsHasMore}
            onLoadMore={handleLoadMoreDatasets}
            onDatasetClick={handleDatasetClick}
          />
        </View>
      )}

      {/* 详情视图 */}
      {state.currentView === 'detail' && state.currentDataset && (
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

            {/* 图片类型筛选 */}
            <View className="filter-section">
              <FilterTabs
                tabs={imageTypeTabs}
                activeKey={state.currentImageType}
                onChange={handleImageTypeFilter}
              />
            </View>

            {/* 图片网格 */}
            <View className="images-section">
              {state.images.length === 0 && !state.imagesLoading ? (
                <EmptyState type="image" title="暂无图片" description="该数据集中没有符合条件的图片" />
              ) : (
                <ImageGrid
                  images={state.images}
                  loading={state.imagesLoading}
                  onImageClick={handleImageClick}
                />
              )}
            </View>

            {/* 加载更多触发器 */}
            {state.imagesHasMore && (
              <View className="load-more-section">
                <View className="load-more-trigger" onClick={handleLoadMoreImages}>
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
  )
}

// 包装组件
const DatasetPage: React.FC = () => {
  return (
    <DatasetProvider>
      <DatasetContent />
    </DatasetProvider>
  )
}

export default DatasetPage