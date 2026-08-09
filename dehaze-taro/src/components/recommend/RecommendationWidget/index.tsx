import React, { useState } from "react";
import { View, Image, Text, ScrollView } from "@tarojs/components";
import Taro from "@tarojs/taro";
import { Photograph, Star } from "@taroify/icons";
import { Button, Tag, Progress, Skeleton } from "@taroify/core";
import { RecommendationAPI } from "dehaze-sdk-js";
import type { ImageFeatureAnalysis, RecommendedAlgorithm } from "dehaze-sdk-js";
import "./index.less";

interface RecommendationWidgetProps {
  /** 初始图片 URL（可选） */
  imageUrl?: string;
  /** 选择算法后的回调 */
  onSelect?: (algorithm: RecommendedAlgorithm) => void;
  /** 容器宽度，用于图片缩放 */
  containerWidth?: number;
}

const SCENE_LABELS: Record<string, string> = {
  outdoor: "户外",
  indoor: "室内",
  landscape: "风景",
  portrait: "人像",
  urban: "城市",
  nature: "自然",
  other: "其他",
};

const LIGHTING_LABELS: Record<string, string> = {
  bright: "明亮",
  dim: "昏暗",
  backlight: "逆光",
  soft: "柔和",
  harsh: "强烈",
  even: "均匀",
  other: "其他",
};

const COMPLEXITY_LABELS: Record<string, string> = {
  simple: "简单",
  moderate: "中等",
  complex: "复杂",
  dense: "密集",
  sparse: "稀疏",
  other: "其他",
};

const HAZE_LEVEL_COLORS: Record<
  string,
  "success" | "warning" | "danger" | "info" | "default"
> = {
  clear: "success",
  light: "warning",
  moderate: "danger",
  heavy: "info",
  severe: "danger",
  other: "default",
};

const RecommendationWidget: React.FC<RecommendationWidgetProps> = ({
  imageUrl: propsImageUrl,
  onSelect,
  containerWidth = 375,
}) => {
  const [selectedImageUri, setSelectedImageUri] = useState<string | null>(
    propsImageUrl || null
  );
  const [analyzing, setAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState<ImageFeatureAnalysis | null>(null);
  const [recommendations, setRecommendations] = useState<
    RecommendedAlgorithm[]
  >([]);

  // 选择图片
  const handleChooseImage = async () => {
    try {
      const res = await Taro.chooseImage({
        count: 1,
        sizeType: ["compressed"],
      });
      if (res.tempFilePaths[0]) {
        setSelectedImageUri(res.tempFilePaths[0]);
        setAnalysis(null);
        setRecommendations([]);
      }
    } catch {
      Taro.showToast({ title: "取消选择", icon: "none" });
    }
  };

  // 分析图像并推荐算法
  const handleAnalyze = async () => {
    if (!selectedImageUri) return;
    setAnalyzing(true);
    setAnalysis(null);
    setRecommendations([]);
    try {
      const result = await RecommendationAPI.analyze({
        imageUrl: selectedImageUri,
      });
      setAnalysis(result);
      const recs = await RecommendationAPI.getAlgorithmRecommendations({
        imageMd5: result.imageMd5,
      });
      setRecommendations(recs || []);
    } catch {
      Taro.showToast({ title: "分析失败", icon: "none" });
    } finally {
      setAnalyzing(false);
    }
  };

  // 重置
  const handleReset = () => {
    setSelectedImageUri(propsImageUrl || null);
    setAnalysis(null);
    setRecommendations([]);
  };

  // 处理选择算法
  const handleSelectAlgorithm = (rec: RecommendedAlgorithm) => {
    onSelect?.(rec);
    Taro.showToast({
      title: `已选择「${rec.algorithmName}」`,
      icon: "success",
    });
  };

  // 无图片状态
  if (!selectedImageUri) {
    return (
      <View className="rec-widget">
        <View className="rec-header">
          <Text className="rec-title">智能算法推荐</Text>
          <Text className="rec-desc">
            上传一张图片，AI 自动分析并推荐最佳去雾算法
          </Text>
        </View>
        <View className="rec-empty-area" onClick={handleChooseImage}>
          <Photograph size="48" color="var(--color-text-muted)" />
          <Text className="rec-empty-text">点击选择图片</Text>
        </View>
      </View>
    );
  }

  return (
    <View className="rec-widget">
      {/* 顶部：图片预览 + 操作 */}
      <View className="rec-preview-section">
        <View className="rec-image-wrapper">
          <Image
            src={selectedImageUri}
            mode="aspectFill"
            style={{ width: containerWidth, height: containerWidth * 0.56 }}
          />
        </View>
        <View className="rec-action-bar">
          <Button
            variant="outlined"
            size="small"
            onClick={handleChooseImage}
            className="rec-action-btn"
          >
            <Photograph size="14" color="var(--color-primary)" />
            <Text className="rec-action-text">换图</Text>
          </Button>
          {!analysis && !analyzing && (
            <Button
              variant="contained"
              size="small"
              onClick={handleAnalyze}
              className="rec-action-btn"
            >
              <Star size="14" color="var(--color-text-inverse)" />
              <Text className="rec-action-text">开始分析</Text>
            </Button>
          )}
          {analyzing && <View className="rec-loading-tip">分析中...</View>}
          {analysis && (
            <Button
              variant="outlined"
              size="small"
              onClick={handleReset}
              className="rec-action-btn"
            >
              <Text className="rec-action-text">重新开始</Text>
            </Button>
          )}
        </View>
      </View>

      {/* 特征分析结果 */}
      {analysis && (
        <ScrollView className="rec-scroll" scrollY>
          <View className="rec-analysis-section">
            <Text className="rec-section-title">图像特征分析</Text>

            <View className="rec-feature-row">
              <Text className="rec-feature-label">雾霾程度</Text>
              <View className="rec-feature-value">
                <Tag
                  color={
                    HAZE_LEVEL_COLORS[analysis.hazeLevel] ||
                    HAZE_LEVEL_COLORS.other
                  }
                  size="small"
                >
                  {analysis.hazeLevel?.replace(/_/g, " ") || analysis.hazeLevel}
                </Tag>
              </View>
            </View>

            <View className="rec-feature-row">
              <Text className="rec-feature-label">场景类型</Text>
              <View className="rec-feature-value">
                <Tag color="primary" size="small">
                  {SCENE_LABELS[
                    analysis.sceneType as keyof typeof SCENE_LABELS
                  ] || analysis.sceneType}
                </Tag>
              </View>
            </View>

            <View className="rec-feature-row">
              <Text className="rec-feature-label">光照条件</Text>
              <View className="rec-feature-value">
                <Tag color="warning" size="small">
                  {LIGHTING_LABELS[
                    analysis.lighting as keyof typeof LIGHTING_LABELS
                  ] || analysis.lighting}
                </Tag>
              </View>
            </View>

            <View className="rec-feature-row">
              <Text className="rec-feature-label">复杂度</Text>
              <View className="rec-feature-value">
                <Tag color="danger" size="small">
                  {COMPLEXITY_LABELS[String(analysis.complexity)] ||
                    analysis.complexity}
                </Tag>
              </View>
            </View>
          </View>

          {/* 推荐算法列表 */}
          {recommendations.length > 0 && (
            <View className="rec-rec-section">
              <Text className="rec-section-title">
                推荐算法 Top {recommendations.length}
              </Text>

              {recommendations.slice(0, 5).map((rec) => (
                <View key={rec.algorithmId} className="rec-algo-card">
                  <View className="rec-algo-header">
                    <View className="rec-algo-name-row">
                      <Text className="rec-algo-name">{rec.algorithmName}</Text>
                      <View className="rec-match-badge">
                        <Text className="rec-match-score">
                          {(rec.matchScore * 100).toFixed(0)}%
                        </Text>
                      </View>
                    </View>
                    <Progress
                      percent={Number((rec.matchScore * 100).toFixed(0))}
                    />
                  </View>
                  {rec.reason && (
                    <Text className="rec-algo-reason">{rec.reason}</Text>
                  )}
                  <View className="rec-algo-actions">
                    <Button
                      variant="contained"
                      size="small"
                      onClick={() => handleSelectAlgorithm(rec)}
                    >
                      选择
                    </Button>
                  </View>
                </View>
              ))}
            </View>
          )}
        </ScrollView>
      )}

      {/* 分析中骨架屏 */}
      {analyzing && (
        <View className="rec-skeleton">
          <Skeleton row={1} title />
        </View>
      )}
    </View>
  );
};

export default RecommendationWidget;
