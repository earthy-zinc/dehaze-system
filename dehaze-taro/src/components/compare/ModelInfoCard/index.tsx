import React from "react";
import { View, Text } from "@tarojs/components";
import type { ImageTaskResponse } from "dehaze-sdk-js";
import type { SelectedModel } from "@/stores/process";
import { formatSeconds } from "@/utils/format";
import "./index.less";

interface ModelInfoCardProps {
  model: SelectedModel | null;
  result: ImageTaskResponse | null;
  /** 评估耗时（可选，来自 metrics 页） */
  evaluationTime?: number;
}

/**
 * 模型信息展示卡片（统一组件）
 * 包含：模型基本信息 + 性能数据
 */
const ModelInfoCard: React.FC<ModelInfoCardProps> = ({
  model,
  result,
  evaluationTime,
}) => {
  if (!model && !result) return null;

  return (
    <View className="model-info-card">
      {model && (
        <View className="info-section">
          <View className="section-title">
            <Text>模型信息</Text>
          </View>
          <View className="info-row">
            <Text className="info-label">模型名称</Text>
            <Text className="info-value">{model.displayName}</Text>
          </View>
          <View className="info-row">
            <Text className="info-label">模型标识</Text>
            <Text className="info-value">{model.modelId}</Text>
          </View>
        </View>
      )}

      {result && (
        <View className="info-section">
          <View className="section-title">
            <Text>性能数据</Text>
          </View>
          <View className="info-row">
            <Text className="info-label">处理耗时</Text>
            <Text className="info-value">
              {formatSeconds(result.time ?? 0)}
            </Text>
          </View>
          {evaluationTime !== undefined && (
            <View className="info-row">
              <Text className="info-label">评估耗时</Text>
              <Text className="info-value">
                {formatSeconds(evaluationTime)}
              </Text>
            </View>
          )}
        </View>
      )}
    </View>
  );
};

export default ModelInfoCard;
