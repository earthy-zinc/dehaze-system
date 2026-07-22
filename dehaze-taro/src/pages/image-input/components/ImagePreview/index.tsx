/**
 * 图片预览组件
 */

import React from "react";
import { View, Text, Image } from "@tarojs/components";
import { Popup, Button } from "@taroify/core";
import { Cross, PhotoOutlined, Expand } from "@taroify/icons";
import { ImageData } from "../../services/types";
import { formatFileSize } from "../../services/imageInput";
import "./ImagePreview.less";

interface ImagePreviewProps {
  visible: boolean;
  imageData: ImageData | null;
  onConfirm: () => void;
  onCancel: () => void;
}

const ImagePreview: React.FC<ImagePreviewProps> = ({
  visible,
  imageData,
  onConfirm,
  onCancel,
}) => {
  if (!imageData) return null;

  return (
    <Popup
      open={visible}
      placement="bottom"
      rounded
      style={{ height: "85vh" }}
      onClose={onCancel}
    >
      <View className="image-preview">
        {/* 头部 */}
        <View className="preview-header">
          <Text className="preview-title">图片预览</Text>
          <View className="close-btn" onClick={onCancel}>
            <Cross size="20" />
          </View>
        </View>

        {/* 图片展示区 */}
        <View className="preview-image-container">
          <Image
            className="preview-image"
            src={imageData.url}
            mode="aspectFit"
            showMenuByLongpress
          />
          {imageData.sampleInfo && (
            <View className="sample-badge">
              <Text>样例图片</Text>
            </View>
          )}
        </View>

        {/* 图片信息 */}
        <View className="preview-info">
          <View className="info-item">
            <View className="info-icon">
              <PhotoOutlined size="16" color="#3b82f6" />
            </View>
            <Text className="info-label">文件大小</Text>
            <Text className="info-value">{formatFileSize(imageData.size)}</Text>
          </View>
          <View className="info-item">
            <View className="info-icon">
              <Expand size="16" color="#10b981" />
            </View>
            <Text className="info-label">图片尺寸</Text>
            <Text className="info-value">
              {imageData.width > 0
                ? `${imageData.width} × ${imageData.height}`
                : "-"}
            </Text>
          </View>
          {imageData.compressed && (
            <View className="compress-hint">
              <Text>
                图片已自动压缩（原始:{" "}
                {formatFileSize(imageData.originalSize || 0)}）
              </Text>
            </View>
          )}
          {imageData.sampleInfo && (
            <View className="sample-info">
              <View className="sample-row">
                <Text className="sample-label">场景类型</Text>
                <Text className="sample-value">
                  {imageData.sampleInfo.sceneType || "未标注"}
                </Text>
              </View>
              {imageData.sampleInfo.hazeLevel && (
                <View className="sample-row">
                  <Text className="sample-label">雾霾程度</Text>
                  <Text className="sample-value">
                    {imageData.sampleInfo.hazeLevel === "light"
                      ? "轻度"
                      : imageData.sampleInfo.hazeLevel === "medium"
                        ? "中度"
                        : "重度"}
                  </Text>
                </View>
              )}
              {imageData.sampleInfo.recommendAlgorithm && (
                <View className="sample-row">
                  <Text className="sample-label">推荐算法</Text>
                  <Text className="sample-value recommend">
                    {imageData.sampleInfo.recommendAlgorithm}
                  </Text>
                </View>
              )}
            </View>
          )}
        </View>

        {/* 底部操作按钮 */}
        <View className="preview-actions">
          <Button
            className="action-btn cancel"
            variant="outlined"
            onClick={onCancel}
          >
            重新选择
          </Button>
          <Button
            className="action-btn confirm"
            color="primary"
            onClick={onConfirm}
          >
            下一步：选择算法
          </Button>
        </View>
      </View>
    </Popup>
  );
};

export default ImagePreview;
