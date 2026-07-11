import React, { useCallback } from 'react'
import { View } from '@tarojs/components'
import type { ImageUrlVO } from '../../services/types'
import ImageCard from '@/components/common/ImageCard'
import ImageViewer from '@/components/common/ImageViewer'
import './ImageGrid.less'

interface ImageGridProps {
  images: ImageUrlVO[]
  className?: string
  onImageClick?: (image: ImageUrlVO) => void
}

const ImageGrid: React.FC<ImageGridProps> = ({
  images,
  className = '',
  onImageClick,
}) => {
  const [viewerImage, setViewerImage] = React.useState<ImageUrlVO | null>(null)

  const handleImageClick = useCallback((image: ImageUrlVO) => {
    setViewerImage(image)
    onImageClick?.(image)
  }, [onImageClick])

  const handleCloseViewer = useCallback(() => {
    setViewerImage(null)
  }, [])

  return (
    <>
      <View className={`image-grid ${className}`}>
        {images.map((image) => (
          <View key={image.id} className="grid-item">
            <ImageCard
              src={image.url}
              filename={image.fileName}
              imageType={image.type as 'clear' | 'hazy'}
              width={image.width}
              height={image.height}
              fileSize={image.sizeBytes}
              onClick={() => handleImageClick(image)}
            />
          </View>
        ))}
      </View>

      <ImageViewer
        visible={!!viewerImage}
        src={viewerImage?.url || ''}
        filename={viewerImage?.fileName}
        imageType={viewerImage?.type as 'clear' | 'hazy' | undefined}
        width={viewerImage?.width}
        height={viewerImage?.height}
        fileSize={viewerImage?.sizeBytes}
        description={viewerImage?.description}
        onClose={handleCloseViewer}
      />
    </>
  )
}

export default ImageGrid
