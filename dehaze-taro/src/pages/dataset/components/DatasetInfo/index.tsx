import React from 'react'
import { View, Text } from '@tarojs/components'
import { Dataset } from '../../services/types'
import './DatasetInfo.less'

interface DatasetInfoProps {
  dataset: Dataset
  className?: string
}

const DatasetInfo: React.FC<DatasetInfoProps> = ({
  dataset,
  className = '',
}) => {
  return (
    <View className={`dataset-info ${className}`}>
      <View className="info-header">
        <Text className="dataset-title">{dataset.name}</Text>
        <Text className="dataset-subtitle">{dataset.description || '暂无描述'}</Text>
      </View>

      <View className="stats-grid">
        <View className="stat-box">
          <Text className="stat-value">{dataset.total_images.toLocaleString()}</Text>
          <Text className="stat-label">总计图片</Text>
        </View>
        <View className="stat-box">
          <Text className="stat-value">{dataset.foggy_count.toLocaleString()}</Text>
          <Text className="stat-label">有雾图片</Text>
        </View>
        <View className="stat-box">
          <Text className="stat-value">{dataset.clear_count.toLocaleString()}</Text>
          <Text className="stat-label">无雾图片</Text>
        </View>
        <View className="stat-box">
          <Text className="stat-value">{dataset.annotated_count.toLocaleString()}</Text>
          <Text className="stat-label">标注图片</Text>
        </View>
      </View>

      <View className="meta-info">
        <View className="meta-item">
          <Text className="meta-label">创建者:</Text>
          <Text className="meta-value">{dataset.creator}</Text>
        </View>
        <View className="meta-item">
          <Text className="meta-label">创建时间:</Text>
          <Text className="meta-value">{new Date(dataset.created_at).toLocaleDateString('zh-CN')}</Text>
        </View>
        <View className="meta-item">
          <Text className="meta-label">更新时间:</Text>
          <Text className="meta-value">{new Date(dataset.updated_at).toLocaleDateString('zh-CN')}</Text>
        </View>
      </View>
    </View>
  )
}

export default DatasetInfo