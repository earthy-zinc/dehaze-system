import { RecommendationAPI } from "dehaze-sdk-js";
import type { ImageFeatureAnalysis, RecommendedAlgorithm } from "dehaze-sdk-js";
import Taro from "@tarojs/taro";
import { useState, useCallback } from "react";

/**
 * 图像算法推荐 Hook
 */
export function useRecommendation() {
  const [analyzing, setAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState<ImageFeatureAnalysis | null>(null);
  const [recommendations, setRecommendations] = useState<
    RecommendedAlgorithm[]
  >([]);

  const analyze = useCallback(async (imageUrl: string) => {
    setAnalyzing(true);
    setAnalysis(null);
    setRecommendations([]);
    try {
      const result = await RecommendationAPI.analyze({ imageUrl });
      setAnalysis(result);
      const recs = await RecommendationAPI.getAlgorithmRecommendations({
        imageMd5: result.imageMd5,
      });
      setRecommendations(recs || []);
    } catch {
      Taro.showToast({ title: "推荐分析失败", icon: "none" });
    } finally {
      setAnalyzing(false);
    }
  }, []);

  const submitFeedback = useCallback(
    async (recommendationId: number, useful: boolean) => {
      try {
        await RecommendationAPI.submitFeedback({ recommendationId, useful });
      } catch {
        // 静默失败
      }
    },
    []
  );

  const reset = useCallback(() => {
    setAnalysis(null);
    setRecommendations([]);
  }, []);

  return {
    analyzing,
    analysis,
    recommendations,
    analyze,
    submitFeedback,
    reset,
  };
}
