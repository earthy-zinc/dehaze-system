import React from "react";
import { View, Text, Image, ScrollView } from "@tarojs/components";
import { Popup, Button } from "@taroify/core";
import type { AlgorithmDetailVO } from "dehaze-sdk-js";

interface AlgorithmDetailPopupProps {
  algorithm: AlgorithmDetailVO | null;
  isFavorite: boolean;
  loading: boolean;
  testResult: string | null;
  testLoading: boolean;
  hasImage: boolean;
  onClose: () => void;
  onToggleFavorite: () => void;
  onSelect: () => void;
  onCustomTest: () => void;
}

const AlgorithmDetailPopup: React.FC<AlgorithmDetailPopupProps> = ({
  algorithm,
  isFavorite,
  loading,
  testResult,
  testLoading,
  hasImage,
  onClose,
  onToggleFavorite,
  onSelect,
  onCustomTest,
}) => {
  return (
    <Popup
      open={!!algorithm || loading}
      placement="bottom"
      style={{ height: "80%", borderRadius: "32rpx 32rpx 0 0" }}
      onClose={onClose}
    >
      {loading ? (
        <View className="loading-state">
          <Text>加载中...</Text>
        </View>
      ) : algorithm ? (
        <View className="detail-popup">
          <View className="detail-header">
            <Text className="detail-title">算法详情</Text>
            <View className="detail-close" onClick={onClose}>
              <Text>✕</Text>
            </View>
          </View>
          <ScrollView className="detail-body" scrollY>
            {/* 基本信息 */}
            <View className="detail-section">
              <Text className="detail-section-title">基本信息</Text>
              <View className="detail-item">
                <Text className="detail-label">算法名称</Text>
                <Text className="detail-value">{algorithm.name}</Text>
              </View>
              {algorithm.type && (
                <View className="detail-item">
                  <Text className="detail-label">算法类型</Text>
                  <Text className="detail-value">{algorithm.type}</Text>
                </View>
              )}
              {algorithm.version && (
                <View className="detail-item">
                  <Text className="detail-label">版本</Text>
                  <Text className="detail-value">{algorithm.version}</Text>
                </View>
              )}
              {algorithm.avgRating !== undefined && (
                <View className="detail-item">
                  <Text className="detail-label">评分</Text>
                  <Text className="detail-value">
                    {algorithm.avgRating} / 5 ({algorithm.ratingCount || 0} 评价)
                  </Text>
                </View>
              )}
              {algorithm.usageCount !== undefined && (
                <View className="detail-item">
                  <Text className="detail-label">使用次数</Text>
                  <Text className="detail-value">{algorithm.usageCount}</Text>
                </View>
              )}
            </View>

            {/* 算法描述 */}
            {algorithm.description && (
              <View className="detail-section">
                <Text className="detail-section-title">算法描述</Text>
                <Text className="detail-desc">{algorithm.description}</Text>
              </View>
            )}

            {/* 性能指标 */}
            {(algorithm.size || algorithm.params || algorithm.flops) && (
              <View className="detail-section">
                <Text className="detail-section-title">性能指标</Text>
                {algorithm.size && (
                  <View className="detail-item">
                    <Text className="detail-label">模型大小</Text>
                    <Text className="detail-value">{algorithm.size}</Text>
                  </View>
                )}
                {algorithm.params && (
                  <View className="detail-item">
                    <Text className="detail-label">参数量</Text>
                    <Text className="detail-value">{algorithm.params}</Text>
                  </View>
                )}
                {algorithm.flops && (
                  <View className="detail-item">
                    <Text className="detail-label">FLOPs</Text>
                    <Text className="detail-value">{algorithm.flops}</Text>
                  </View>
                )}
              </View>
            )}

            {/* 效果样例 */}
            {algorithm.sampleImages && algorithm.sampleImages.length > 0 && (
              <View className="detail-section">
                <Text className="detail-section-title">效果样例</Text>
                <ScrollView className="sample-scroll" scrollX>
                  <View className="sample-list">
                    {algorithm.sampleImages.map((url, idx) => (
                      <Image
                        key={idx}
                        src={url}
                        className="sample-image"
                        mode="aspectFill"
                        lazyLoad
                      />
                    ))}
                  </View>
                </ScrollView>
              </View>
            )}

            {/* 自定义测试 */}
            {hasImage && (
              <View className="detail-section">
                <Text className="detail-section-title">自定义测试</Text>
                {testResult ? (
                  <Image
                    src={testResult}
                    className="test-result-image"
                    mode="widthFix"
                  />
                ) : (
                  <Button
                    size="small"
                    variant="outlined"
                    loading={testLoading}
                    onClick={onCustomTest}
                  >
                    使用当前图片测试效果
                  </Button>
                )}
              </View>
            )}
          </ScrollView>
          <View className="detail-footer">
            <Button variant="outlined" onClick={onToggleFavorite}>
              {isFavorite ? "取消收藏" : "收藏"}
            </Button>
            <Button color="primary" onClick={onSelect}>
              立即使用
            </Button>
          </View>
        </View>
      ) : null}
    </Popup>
  );
};

export default AlgorithmDetailPopup;
