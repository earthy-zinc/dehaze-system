import React, { useCallback } from "react";
import { View } from "@tarojs/components";
import ImageCard from "@/components/common/ImageCard";
import ImageViewer from "@/components/common/ImageViewer";
import type { ImageUrlVO } from "../../services/types";
import "./ImageGrid.less";

interface ImageGridProps {
  images: ImageUrlVO[];
  className?: string;
}

const ImageGrid: React.FC<ImageGridProps> = ({ images, className = "" }) => {
  const [viewerImage, setViewerImage] = React.useState<ImageUrlVO | null>(null);

  const handleImageClick = useCallback((image: ImageUrlVO) => {
    setViewerImage(image);
  }, []);

  const handleCloseViewer = useCallback(() => {
    setViewerImage(null);
  }, []);

  return (
    <>
      <View className={`image-grid ${className}`}>
        {images.map((image) => (
          <View key={image.id} className="grid-item">
            <ImageCard
              src={image.url}
              filename={image.fileName}
              imageType={image.type}
              hazeLevel={image.hazeLevel}
              width={image.width}
              height={image.height}
              fileSize={image.sizeBytes}
              onClick={() => handleImageClick(image)}
            />
          </View>
        ))}
      </View>

      {viewerImage && (
        <ImageViewer
          visible={true}
          src={viewerImage.url}
          filename={viewerImage.fileName}
          imageType={viewerImage.type}
          hazeLevel={viewerImage.hazeLevel}
          width={viewerImage.width}
          height={viewerImage.height}
          fileSize={viewerImage.sizeBytes}
          onClose={handleCloseViewer}
        />
      )}
    </>
  );
};

export default ImageGrid;
