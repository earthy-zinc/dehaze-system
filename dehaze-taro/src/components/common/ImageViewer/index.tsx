import React, { useEffect } from "react";
import { View, Image, Text } from "@tarojs/components";
import { Close } from "@taroify/icons";
import {
  getImageTypeLabel,
  formatHazeLevel,
} from "@/pages/dataset/services/imageUtils";
import "./ImageViewer.less";

export interface ImageViewerProps {
  visible: boolean;
  src: string;
  filename?: string;
  /** 图片类型：clear/hazy/trans/depth/segment */
  imageType?: string;
  /** 雾霾程度：light/medium/heavy、beta=0.5 等，可为空（表示未标注） */
  hazeLevel?: string;
  width?: number;
  height?: number;
  fileSize?: number;
  tags?: string;
  description?: string;
  onClose: () => void;
}

const ImageViewer: React.FC<ImageViewerProps> = ({
  visible,
  src,
  filename,
  imageType,
  hazeLevel,
  width,
  height,
  fileSize,
  tags,
  description,
  onClose,
}) => {
  // 防止背景滚动 (仅在H5环境生效)
  useEffect(() => {
    if (typeof window !== "undefined" && typeof document !== "undefined") {
      if (visible) {
        document.body.style.overflow = "hidden";
      } else {
        document.body.style.overflow = "auto";
      }

      return () => {
        document.body.style.overflow = "auto";
      };
    }
  }, [visible]);

  if (!visible) return null;

  const typeLabel = getImageTypeLabel(imageType) || "未知类型";
  const hazeLevelLabel = formatHazeLevel(hazeLevel);

  const formatFileSize = (bytes?: number) => {
    if (!bytes) return "-";
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  };

  const handleMaskClick = (e: any) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  return (
    <View className="image-viewer" onClick={handleMaskClick}>
      <View className="viewer-container">
        <View className="close-btn" onClick={onClose}>
          <Close size="24" color="white" />
        </View>

        <View className="image-container">
          <Image src={src} className="viewer-image" mode="aspectFit" />
        </View>

        <View className="image-details">
          {filename && <Text className="detail-filename">{filename}</Text>}

          <View className="detail-meta">
            {imageType && (
              <View className="meta-item">
                <Text className="meta-label">类型:</Text>
                <Text className="meta-value">{typeLabel}</Text>
              </View>
            )}

            <View className="meta-item">
              <Text className="meta-label">雾霾程度:</Text>
              <Text className="meta-value">{hazeLevelLabel || "未标注"}</Text>
            </View>

            {width && height && (
              <View className="meta-item">
                <Text className="meta-label">尺寸:</Text>
                <Text className="meta-value">
                  {width} × {height}
                </Text>
              </View>
            )}

            {fileSize && (
              <View className="meta-item">
                <Text className="meta-label">大小:</Text>
                <Text className="meta-value">{formatFileSize(fileSize)}</Text>
              </View>
            )}

            {tags && (
              <View className="meta-item">
                <Text className="meta-label">标签:</Text>
                <Text className="meta-value">{tags}</Text>
              </View>
            )}
          </View>

          {description && (
            <Text className="detail-description">{description}</Text>
          )}
        </View>
      </View>
    </View>
  );
};

export default ImageViewer;
