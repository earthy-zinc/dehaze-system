import React from "react";
import { View, Image, Text } from "@tarojs/components";
import {
  getImageTypeLabel,
  getImageTypeBadgeClass,
} from "@/pages/dataset/services/imageUtils";
import { formatFileSize, formatHazeLevel } from "@/utils/format";
import "./ImageCard.less";

export interface ImageCardProps {
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
  className?: string;
  onClick?: () => void;
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
  className = "",
  onClick,
}) => {
  const typeLabel = getImageTypeLabel(imageType);
  const typeBadgeClass = getImageTypeBadgeClass(imageType);
  const hazeLevelLabel = formatHazeLevel(hazeLevel);

  return (
    <View className={`image-card ${className}`} onClick={onClick}>
      <View className="image-wrapper">
        <Image src={src} className="image" mode="aspectFill" lazyLoad />
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
        {filename && <Text className="image-filename">{filename}</Text>}
        {width && height && (
          <Text className="image-meta">
            {width} × {height}
          </Text>
        )}
        {fileSize && (
          <Text className="image-meta">{formatFileSize(fileSize)}</Text>
        )}
        {tags && (
          <View className="image-tags">
            {tags.split(",").map((tag) => (
              <Text key={tag} className="tag">
                {tag.trim()}
              </Text>
            ))}
          </View>
        )}
      </View>
    </View>
  );
};

export default ImageCard;
