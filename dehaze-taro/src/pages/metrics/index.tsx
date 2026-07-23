import React, { useState, useEffect, useCallback } from "react";
import { View, Text, ScrollView } from "@tarojs/components";
import CompareNavbar from "@/components/compare/CompareNavbar";
import { ModelAPI } from "dehaze-sdk-js";
import type { EvaluationResultVO } from "dehaze-sdk-js";
import CompareToolbar from "@/components/compare/CompareToolbar";
import AlgorithmInfoCard from "@/components/compare/AlgorithmInfoCard";
import { loadCompareContext } from "@/components/compare/types";
import "./index.less";

// 指标中文说明
const METRIC_LABELS: Record<
  string,
  { label: string; unit: string; better: "higher" | "lower"; desc: string }
> = {
  psnr: {
    label: "峰值信噪比",
    unit: "dB",
    better: "higher",
    desc: "越高越好，通常>30dB为好",
  },
  ssim: {
    label: "结构相似性",
    unit: "",
    better: "higher",
    desc: "越接近1越好，>0.85为好",
  },
  lpips: {
    label: "感知相似度",
    unit: "",
    better: "lower",
    desc: "越低越好，<0.3为好",
  },
  niqe: {
    label: "自然图像质量",
    unit: "",
    better: "lower",
    desc: "越低越好，<5为好",
  },
  entropy: {
    label: "信息熵",
    unit: "",
    better: "higher",
    desc: "越高越好，7-8为佳",
  },
  mse: { label: "均方误差", unit: "", better: "lower", desc: "越小越好" },
};

const MetricsPage: React.FC = () => {
  const [ctx] = useState(loadCompareContext);
  const [evaluation, setEvaluation] = useState<EvaluationResultVO | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const { algorithm, result: prediction, originImage } = ctx;
  const cleanUrl = originImage?.cleanUrl;

  // 执行评估（GT 必须是 clean 图，不能用 hazy 原图）
  const fetchEvaluation = useCallback(async () => {
    if (!algorithm || !prediction?.resultUrl) {
      setLoading(false);
      return;
    }

    if (!cleanUrl) {
      setLoading(false);
      setError("该图片无GT参考，无法评估");
      return;
    }

    try {
      setLoading(true);
      setError("");

      const res = await ModelAPI.evaluate({
        algorithmId: algorithm.id,
        predUrl: prediction.resultUrl,
        gtUrl: cleanUrl,
      });
      setEvaluation(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "评估失败，可能需要参考图像(GT)");
    } finally {
      setLoading(false);
    }
  }, [algorithm, prediction, cleanUrl]);

  useEffect(() => {
    fetchEvaluation();
  }, [fetchEvaluation]);

  // 格式化指标值
  const formatMetric = (key: string, value: number) => {
    const config = METRIC_LABELS[key];
    if (!config) return value.toFixed(2);
    if (key === "psnr") return value.toFixed(2) + " " + config.unit;
    if (key === "ssim" || key === "lpips") return value.toFixed(4);
    return value.toFixed(2) + (config.unit ? " " + config.unit : "");
  };

  // 获取指标状态颜色
  const getMetricStatus = (
    key: string,
    value: number
  ): "good" | "normal" | "bad" => {
    const config = METRIC_LABELS[key];
    if (!config) return "normal";
    const thresholds: Record<string, [number, number]> = {
      psnr: [30, 25],
      ssim: [0.85, 0.7],
      lpips: [0.3, 0.5],
      niqe: [5, 8],
    };
    const range = thresholds[key];
    if (!range) return "normal";
    if (config.better === "higher") {
      return value >= range[0] ? "good" : value >= range[1] ? "normal" : "bad";
    } else {
      return value <= range[0] ? "good" : value <= range[1] ? "normal" : "bad";
    }
  };

  const metrics = evaluation?.metrics || {};
  const metricKeys = Object.keys(metrics);

  return (
    <View className="metrics-page">
      {/* 顶部导航 */}
      <CompareNavbar title="指标对比" />

      <ScrollView className="metrics-content" scrollY>
        {/* 算法信息 + 性能数据（使用统一组件，含评估耗时） */}
        <AlgorithmInfoCard
          algorithm={algorithm}
          result={prediction}
          evaluationTime={evaluation?.time}
        />

        {/* 评估指标 */}
        <View className="info-card">
          <View className="card-title">
            <Text>质量评估指标</Text>
            {evaluation?.qualified !== undefined && (
              <View
                className={`qualified-tag ${evaluation.qualified ? "qualified" : "unqualified"}`}
              >
                <Text>{evaluation.qualified ? "合格" : "不合格"}</Text>
              </View>
            )}
          </View>

          {loading ? (
            <View className="loading-state">
              <View className="loading-spinner" />
              <Text>正在计算评估指标...</Text>
            </View>
          ) : error ? (
            <View className="error-state">
              <Text className="error-text">{error}</Text>
              <View className="retry-btn" onClick={fetchEvaluation}>
                <Text>重试</Text>
              </View>
            </View>
          ) : metricKeys.length === 0 ? (
            <View className="empty-metrics">
              <Text>暂无评估数据</Text>
            </View>
          ) : (
            <View className="metrics-list">
              {metricKeys.map((key) => {
                const value = metrics[key];
                const config = METRIC_LABELS[key];
                const status = getMetricStatus(key, value);
                return (
                  <View key={key} className="metric-item">
                    <View className="metric-info">
                      <Text className="metric-label">
                        {config?.label || key}
                      </Text>
                      <Text className="metric-desc">{config?.desc || ""}</Text>
                    </View>
                    <View className="metric-right">
                      <Text className={`metric-value metric-${status}`}>
                        {formatMetric(key, value)}
                      </Text>
                      {config?.better === "higher" ? (
                        <Text className="metric-arrow">↑</Text>
                      ) : config?.better === "lower" ? (
                        <Text className="metric-arrow">↓</Text>
                      ) : null}
                    </View>
                  </View>
                );
              })}
            </View>
          )}
        </View>
      </ScrollView>

      {/* 底部工具栏 */}
      <CompareToolbar currentMode="metrics" resultUrl={prediction?.resultUrl} />
    </View>
  );
};

export default MetricsPage;
