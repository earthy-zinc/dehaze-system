import React from 'react'
import { View, Image, Text } from '@tarojs/components'
import {
  getImageTypeLabel,
  getImageTypeBadgeClass,
  formatHazeLevel,
} from '@/pages/dataset/services/imageUtils'
import './ImageCard.less'

export interface ImageCardProps {
  src: string
  filename?: string
  /** 图片类型：clear/hazy/trans/depth/segment */
  imageType?: string
  /** 雾霾程度：light/medium/heavy、beta=0.5 等，可为空（表示未标注） */
  hazeLevel?: string
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
  hazeLevel,
  width,
  height,
  fileSize,
  tags,
  className = '',
  onClick,
}) => {
  const typeLabel = getImageTypeLabel(imageType)
  const typeBadgeClass = getImageTypeBadgeClass(imageType)
  const hazeLevelLabel = formatHazeLevel(hazeLevel)

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
          <View className={`type-badge ${typeBadgeClass}`}>
            <Text className="type-text">{typeLabel}</Text>
          </View>
        )}
        {hazeLevelLabel && (
          <View className="haze-badge">
            <Text className="haze-text">{hazeLevelLabel}</Text>
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
