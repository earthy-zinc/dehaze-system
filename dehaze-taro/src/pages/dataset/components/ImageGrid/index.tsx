import React, { useCallback } from 'react'
import { View } from '@tarojs/components'
import { DatasetImage } from '../../services/types'
import ImageCard from '@/components/common/ImageCard'
import ImageViewer from '@/components/common/ImageViewer'
import './ImageGrid.less'

interface ImageGridProps {
  images: DatasetImage[]
  loading?: boolean
  className?: string
  onImageClick?: (image: DatasetImage) => void
}

const ImageGrid: React.FC<ImageGridProps> = ({
  images,
  loading = false,
  className = '',
  onImageClick,
}) => {
  const [viewerImage, setViewerImage] = React.useState<DatasetImage | null>(null)

  const handleImageClick = useCallback((image: DatasetImage) => {
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
              src={image.image_url}
              alt={image.filename}
              filename={image.filename}
              imageType={image.image_type}
              width={image.width}
              height={image.height}
              fileSize={image.file_size}
              tags={image.tags}
              onClick={() => handleImageClick(image)}
            />
          </View>
        ))}
      </View>

      <ImageViewer
        visible={!!viewerImage}
        src={viewerImage?.image_url || ''}
        alt={viewerImage?.filename || ''}
        filename={viewerImage?.filename}
        imageType={viewerImage?.image_type}
        width={viewerImage?.width}
        height={viewerImage?.height}
        fileSize={viewerImage?.file_size}
        tags={viewerImage?.tags}
        description={viewerImage?.description}
        onClose={handleCloseViewer}
      />
    </>
  )
}

export default ImageGrid