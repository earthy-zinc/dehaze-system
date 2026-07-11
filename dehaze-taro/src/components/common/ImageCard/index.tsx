import React from 'react'
import { View, Image, Text } from '@tarojs/components'
import './ImageCard.less'

export interface ImageCardProps {
  src: string
  filename?: string
  imageType?: 'clear' | 'hazy'
  width?: number
  height?: number
  fileSize?: number
  tags?: string
  className?: string
  onClick?: () => void
}

const ImageCard: React.FC<ImageCardProps> = ({
  src,
  filename,
  imageType,
  width,
  height,
  fileSize,
  tags,
  className = '',
  onClick,
}) => {
  const getTypeLabel = (type?: string) => {
    const labels = {
      hazy: '有雾',
      clear: '无雾',
    }
    return labels[type as keyof typeof labels] || ''
  }

  const getTypeClass = (type?: string) => {
    const classes = {
      hazy: 'type-badge-hazy',
      clear: 'type-badge-clear',
    }
    return classes[type as keyof typeof classes] || ''
  }

  const formatFileSize = (bytes?: number) => {
    if (!bytes) return '-'
    if (bytes < 1024) return bytes + ' B'
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
  }

  return (
    <View className={`image-card ${className}`} onClick={onClick}>
      <View className="image-wrapper">
        <Image
          src={src}
          className="image"
          mode="aspectFill"
          lazyLoad
        />
        {imageType && (
          <View className={`type-badge ${getTypeClass(imageType)}`}>
            <Text className="type-text">{getTypeLabel(imageType)}</Text>
          </View>
        )}
      </View>
      <View className="image-info">
        {filename && (
          <Text className="image-filename">{filename}</Text>
        )}
        {(width && height) && (
          <Text className="image-meta">{width} × {height}</Text>
        )}
        {fileSize && (
          <Text className="image-meta">{formatFileSize(fileSize)}</Text>
        )}
        {tags && (
          <View className="image-tags">
            {tags.split(',').map((tag, index) => (
              <Text key={index} className="tag">{tag.trim()}</Text>
            ))}
          </View>
        )}
      </View>
    </View>
  )
}

export default ImageCard