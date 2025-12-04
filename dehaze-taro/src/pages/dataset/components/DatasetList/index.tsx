import React from 'react'
import { View, ScrollView } from '@tarojs/components'
import { Dataset } from '../../services/types'
import DatasetCard from '../DatasetCard'
import EmptyState from '@/components/common/EmptyState'
import './DatasetList.less'

interface DatasetListProps {
  datasets: Dataset[]
  loading?: boolean
  onLoadMore?: () => void
  hasMore?: boolean
  onDatasetClick?: (dataset: Dataset) => void
  className?: string
}

const DatasetList: React.FC<DatasetListProps> = ({
  datasets,
  loading = false,
  onLoadMore,
  hasMore = false,
  onDatasetClick,
  className = '',
}) => {
  const handleScrollToLower = () => {
    if (onLoadMore && hasMore && !loading) {
      onLoadMore()
    }
  }

  if (datasets.length === 0 && !loading) {
    return (
      <View className={`dataset-list ${className}`}>
        <EmptyState type="dataset" />
      </View>
    )
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
          <DatasetCard
            key={dataset.id}
            dataset={dataset}
            onClick={() => onDatasetClick?.(dataset)}
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
  )
}

export default DatasetList