import React from 'react'
import { View, Text, Image } from '@tarojs/components'
import { PhotoOutlined, UserOutlined, CalendarOutlined } from '@taroify/icons'
import { Dataset } from '../../services/types'
import './DatasetCard.less'

interface DatasetCardProps {
  dataset: Dataset
  onClick?: () => void
  className?: string
}

const DatasetCard: React.FC<DatasetCardProps> = ({
  dataset,
  onClick,
  className = '',
}) => {
  const formatDate = (dateString: string) => {
    const date = new Date(dateString)
    const now = new Date()
    const diff = now.getTime() - date.getTime()
    const days = Math.floor(diff / (1000 * 60 * 60 * 24))

    if (days === 0) return '今天'
    if (days === 1) return '昨天'
    if (days < 7) return `${days}天前`

    return date.toLocaleDateString('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
    })
  }

  return (
    <View className={`dataset-card ${className}`} onClick={onClick}>
      <View className="card-content">
        <View className="thumbnail-wrapper">
          <Image
            src={dataset.thumbnail}
            alt={dataset.name}
            className="thumbnail"
            mode="aspectFill"
            lazyLoad
          />
        </View>
        <View className="card-info">
          <Text className="dataset-name">{dataset.name}</Text>
          <Text className="dataset-description">
            {dataset.description || '暂无描述'}
          </Text>
          <View className="dataset-stats">
            <View className="stat-item">
              <PhotoOutlined size='14' color='#9ca3af' />
              <Text className="stat-value">{dataset.total_images}</Text>
            </View>
            <View className="stat-item">
              <UserOutlined size='14' color='#9ca3af' />
              <Text className="stat-value">{dataset.creator}</Text>
            </View>
            <View className="stat-item">
              <CalendarOutlined size='14' color='#9ca3af' />
              <Text className="stat-value">{formatDate(dataset.created_at)}</Text>
            </View>
          </View>
        </View>
      </View>
    </View>
  )
}

export default DatasetCard