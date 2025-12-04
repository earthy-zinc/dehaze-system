import React from 'react'
import { View, Text } from '@tarojs/components'
import './EmptyState.less'

interface EmptyStateProps {
  type?: 'dataset' | 'image' | 'search'
  title?: string
  description?: string
  className?: string
}

const EmptyState: React.FC<EmptyStateProps> = ({
  type = 'dataset',
  title,
  description,
  className = '',
}) => {
  const getDefaultContent = () => {
    switch (type) {
      case 'dataset':
        return {
          icon: '📁',
          title: title || '暂无数据集',
          description: description || '当前没有可用的数据集',
        }
      case 'image':
        return {
          icon: '🖼️',
          title: title || '暂无图片',
          description: description || '该数据集中没有图片',
        }
      case 'search':
        return {
          icon: '🔍',
          title: title || '未找到结果',
          description: description || '请尝试其他搜索关键词',
        }
      default:
        return {
          icon: '📭',
          title: title || '暂无数据',
          description: description || '当前没有可用的数据',
        }
    }
  }

  const content = getDefaultContent()

  return (
    <View className={`empty-state ${className}`}>
      <View className="empty-icon">{content.icon}</View>
      <Text className="empty-title">{content.title}</Text>
      <Text className="empty-description">{content.description}</Text>
    </View>
  )
}

export default EmptyState