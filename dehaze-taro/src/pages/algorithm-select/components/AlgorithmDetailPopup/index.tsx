import React from "react";
import { View, Text, ScrollView } from "@tarojs/components";
import { Popup, Button } from "@taroify/core";
import type { Algorithm } from "dehaze-sdk-js";
import { getStatusInfo, PUBLISHED_STATUS } from "../../utils";

interface AlgorithmDetailPopupProps {
  algorithm: Algorithm | null;
  isFavorite: boolean;
  onClose: () => void;
  onToggleFavorite: (algo: Algorithm) => void;
  onSelect: (algo: Algorithm) => void;
}

const AlgorithmDetailPopup: React.FC<AlgorithmDetailPopupProps> = ({
  algorithm,
  isFavorite,
  onClose,
  onToggleFavorite,
  onSelect,
}) => {
  return (
    <Popup
      open={!!algorithm}
      placement="bottom"
      style={{ height: "60%", borderRadius: "16px 16px 0 0" }}
      onClose={onClose}
    >
      {algorithm && (
        <View className="detail-popup">
          <View className="detail-header">
            <Text className="detail-title">算法详情</Text>
            <View className="detail-close" onClick={onClose}>
              <Text>✕</Text>
            </View>
          </View>
          <ScrollView className="detail-body" scrollY>
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
            {algorithm.size && (
              <View className="detail-item">
                <Text className="detail-label">模型大小</Text>
                <Text className="detail-value">{algorithm.size}</Text>
              </View>
            )}
            {algorithm.flops && (
              <View className="detail-item">
                <Text className="detail-label">计算量</Text>
                <Text className="detail-value">{algorithm.flops}</Text>
              </View>
            )}
            <View className="detail-item">
              <Text className="detail-label">状态</Text>
              <View
                className={`status-tag ${getStatusInfo(algorithm.status).className}`}
              >
                <Text>{getStatusInfo(algorithm.status).label}</Text>
              </View>
            </View>
            {algorithm.description && (
              <View className="detail-item detail-desc-item">
                <Text className="detail-label">描述</Text>
                <Text className="detail-value detail-desc">
                  {algorithm.description}
                </Text>
              </View>
            )}
            {algorithm.createTime && (
              <View className="detail-item">
                <Text className="detail-label">创建时间</Text>
                <Text className="detail-value">{algorithm.createTime}</Text>
              </View>
            )}
          </ScrollView>
          <View className="detail-footer">
            <Button
              variant="outlined"
              onClick={() => onToggleFavorite(algorithm)}
            >
              {isFavorite ? "取消收藏" : "收藏"}
            </Button>
            {algorithm.status === PUBLISHED_STATUS && (
              <Button
                color="primary"
                onClick={() => {
                  onClose();
                  onSelect(algorithm);
                }}
              >
                立即使用
              </Button>
            )}
          </View>
        </View>
      )}
    </Popup>
  );
};

export default AlgorithmDetailPopup;
