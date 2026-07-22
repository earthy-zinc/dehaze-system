/**
 * 上传区域组件
 */

import React from "react";
import { View, Text } from "@tarojs/components";
import { Plus, PhotoOutlined } from "@taroify/icons";
import { Loading } from "@taroify/core";
import "./UploadArea.less";

interface UploadAreaProps {
  onUpload: () => void;
  loading?: boolean;
  error?: string | null;
}

const UploadArea: React.FC<UploadAreaProps> = ({
  onUpload,
  loading = false,
  error,
}) => {
  const handleClick = () => {
    if (!loading) {
      onUpload();
    }
  };

  return (
    <View className="upload-area">
      <View
        className={`upload-zone ${loading ? "loading" : ""}`}
        onClick={handleClick}
      >
        {loading ? (
          <View className="upload-loading">
            <Loading size="32px" />
            <Text className="loading-text">加载中...</Text>
          </View>
        ) : (
          <>
            <View className="upload-icon">
              <PhotoOutlined size="48" color="#9ca3af" />
            </View>
            <View className="upload-content">
              <View className="upload-btn">
                <Plus size="16" />
                <Text>点击上传图片</Text>
              </View>
              <Text className="upload-hint">
                支持 JPG、PNG、WEBP、HEIC 格式
              </Text>
              <Text className="upload-hint">单张图片最大 20MB</Text>
            </View>
          </>
        )}
      </View>

      {error && (
        <View className="upload-error">
          <Text>{error}</Text>
        </View>
      )}

      <View className="upload-tips">
        <View className="tip-item">
          <Text className="tip-icon">💡</Text>
          <Text className="tip-text">大于 5MB 的图片会自动压缩</Text>
        </View>
        <View className="tip-item">
          <Text className="tip-icon">📱</Text>
          <Text className="tip-text">建议上传清晰的雾天照片</Text>
        </View>
      </View>
    </View>
  );
};

export default UploadArea;
