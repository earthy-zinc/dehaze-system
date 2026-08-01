import React from "react";
import { View, Text } from "@tarojs/components";
import type { Algorithm, PredictionResultVO } from "dehaze-sdk-js";
import { formatDuration } from "@/utils/format";
import "./index.less";

interface AlgorithmInfoCardProps {
  algorithm: Algorithm | null;
  result: PredictionResultVO | null;
  /** 评估耗时（可选，来自 metrics 页） */
  evaluationTime?: number;
}

/**
 * 算法信息展示卡片（统一组件）
 * 包含：算法基本信息 + 处理参数 + 性能数据
 */
const AlgorithmInfoCard: React.FC<AlgorithmInfoCardProps> = ({
  algorithm,
  result,
  evaluationTime,
}) => {
  if (!algorithm && !result) return null;

  // 解析算法参数（params 是 string 类型，存的是 JSON）
  let params: Record<string, number> = {};
  if (algorithm?.params) {
    try {
      params = JSON.parse(algorithm.params);
    } catch {
      params = {};
    }
  }

  // 处理参数默认值（去雾强度/饱和度/对比度/锐化）
  const paramConfigs = [
    { key: "strength", label: "去雾强度", default: 50 },
    { key: "saturation", label: "色彩饱和度", default: 50 },
    { key: "contrast", label: "对比度", default: 50 },
    { key: "sharpen", label: "锐化程度", default: 30 },
  ];

  const hasParams = paramConfigs.some((cfg) => params[cfg.key] !== undefined);

  return (
    <View className="algorithm-info-card">
      {/* 算法基本信息 */}
      {algorithm && (
        <View className="info-section">
          <View className="section-title">
            <Text>算法信息</Text>
          </View>
          <View className="info-row">
            <Text className="info-label">算法名称</Text>
            <Text className="info-value">{algorithm.name}</Text>
          </View>
          {algorithm.type && (
            <View className="info-row">
              <Text className="info-label">算法类型</Text>
              <Text className="info-value">{algorithm.type}</Text>
            </View>
          )}
          {algorithm.version && (
            <View className="info-row">
              <Text className="info-label">版本</Text>
              <Text className="info-value">{algorithm.version}</Text>
            </View>
          )}
          {algorithm.description && (
            <View className="info-row">
              <Text className="info-label">适用场景</Text>
              <Text className="info-value">{algorithm.description}</Text>
            </View>
          )}
          {algorithm.flops !== undefined && (
            <View className="info-row">
              <Text className="info-label">计算量</Text>
              <Text className="info-value">{algorithm.flops} GFLOPS</Text>
            </View>
          )}
        </View>
      )}

      {/* 处理参数（含默认值对比） */}
      {hasParams && (
        <View className="info-section">
          <View className="section-title">
            <Text>处理参数</Text>
          </View>
          {paramConfigs.map((cfg) => {
            const value = params[cfg.key];
            if (value === undefined) return null;
            const diff = value - cfg.default;
            return (
              <View key={cfg.key} className="info-row">
                <Text className="info-label">{cfg.label}</Text>
                <View className="param-value">
                  <Text className="info-value">{value}</Text>
                  {diff !== 0 && (
                    <Text
                      className={`param-diff ${diff > 0 ? "positive" : "negative"}`}
                    >
                      ({diff > 0 ? "+" : ""}
                      {diff})
                    </Text>
                  )}
                </View>
              </View>
            );
          })}
        </View>
      )}

      {/* 性能数据 */}
      {result && (
        <View className="info-section">
          <View className="section-title">
            <Text>性能数据</Text>
          </View>
          <View className="info-row">
            <Text className="info-label">处理耗时</Text>
            <Text className="info-value">
              {formatDuration(result.time ?? 0)}
            </Text>
          </View>
          {result.fromCache !== undefined && (
            <View className="info-row">
              <Text className="info-label">缓存命中</Text>
              <Text className="info-value">
                {result.fromCache ? "是" : "否"}
              </Text>
            </View>
          )}
          {evaluationTime !== undefined && (
            <View className="info-row">
              <Text className="info-label">评估耗时</Text>
              <Text className="info-value">
                {formatDuration(evaluationTime)}
              </Text>
            </View>
          )}
        </View>
      )}
    </View>
  );
};

export default AlgorithmInfoCard;
