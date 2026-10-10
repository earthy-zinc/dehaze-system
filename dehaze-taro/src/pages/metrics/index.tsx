import React, { useState, useEffect, useCallback } from "react";
import { View, Text, ScrollView } from "@tarojs/components";
import { Button } from "@taroify/core";
import Taro from "@tarojs/taro";

import ImmersiveLayout from "@/layout/immersive";
import { ModelAPI } from "dehaze-sdk-js";
import type { EvaluationResultVO } from "dehaze-sdk-js";
import CompareToolbar from "@/components/compare/CompareToolbar";
import ModelInfoCard from "@/components/compare/ModelInfoCard";
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

type TaskStatus = 1 | 2 | 3;

const MetricsPage: React.FC = () => {
  const [ctx] = useState(loadCompareContext);
  const [evaluation, setEvaluation] = useState<EvaluationResultVO | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [reportLoading, setReportLoading] = useState(false);
  const [reportDownloading, setReportDownloading] = useState(false);

  const { model, result: prediction, originImage } = ctx;
  const cleanUrl = originImage?.cleanUrl;

  // 执行评估（GT 必须是 clean 图，不能用 hazy 原图）
  const fetchEvaluation = useCallback(async () => {
    if (!model || !prediction?.resultUrl) {
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

      const res = await ModelAPI.evaluateAndWait({
        modelId: model.modelId,
        predUrl: prediction.resultUrl,
        gtUrl: cleanUrl,
      });
      if (res.status === 3) {
        throw new Error(res.errorMessage || "评估失败");
      }
      setEvaluation(res);
    } catch (err: unknown) {
      setError(
        err instanceof Error ? err.message : "评估失败，可能需要参考图像(GT)"
      );
    } finally {
      setLoading(false);
    }
  }, [model, prediction, cleanUrl]);

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

  // 生成并下载报告
  const handleExportReport = async () => {
    if (!prediction?.resultUrl) {
      Taro.showToast({ title: "缺少必要参数", icon: "none" });
      return;
    }
    setReportLoading(true);
    try {
      const res = await ModelAPI.generateReport({
        logId: prediction.taskId ?? 0,
      });
      const taskId = res.taskId;
      if (!taskId) throw new Error("未返回任务ID");
      while (true) {
        const statusRes = await ModelAPI.getReportStatus(taskId);
        const status = statusRes.status as TaskStatus;
        if (status === 2) {
          if (statusRes.downloadUrl) {
            setReportLoading(false);
            setReportDownloading(true);
            try {
              const filePath = await Taro.downloadFile({
                url: statusRes.downloadUrl,
              });
              if (filePath.tempFilePath) {
                await Taro.openDocument({
                  filePath: filePath.tempFilePath,
                  showMenu: true,
                });
              }
            } catch {
              Taro.showToast({ title: "打开报告失败", icon: "none" });
            } finally {
              setReportDownloading(false);
            }
          } else {
            throw new Error("报告生成但无下载链接");
          }
          break;
        }
        if (status === 3)
          throw new Error(statusRes.errorMessage || "报告生成失败");
        await new Promise((r) => setTimeout(r, 2000));
      }
    } catch (err: unknown) {
      Taro.showToast({
        title: err instanceof Error ? err.message : "报告生成失败",
        icon: "none",
      });
    } finally {
      setReportLoading(false);
    }
  };

  const metrics = evaluation?.metrics || {};
  const metricKeys = Object.keys(metrics);

  return (
    <ImmersiveLayout
      title="指标对比"
      toolbar={
        <CompareToolbar
          currentMode="metrics"
          resultUrl={prediction?.resultUrl}
          resultId={prediction?.taskId}
        />
      }
    >
      <ScrollView className="metrics-content" scrollY>
        {/* 模型信息 + 性能数据（使用统一组件，含评估耗时） */}
        <ModelInfoCard
          model={model}
          result={prediction}
          evaluationTime={evaluation?.time}
        />

        {/* 评估指标 */}
        <View className="info-card">
          <View className="card-title">
            <Text>质量评估指标</Text>
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

        {/* 导出报告 */}
        <View className="export-report-section">
          <Button
            block
            color="primary"
            loading={reportLoading || reportDownloading}
            onClick={handleExportReport}
          >
            {reportDownloading ? "正在打开报告..." : "导出报告"}
          </Button>
        </View>
      </ScrollView>
    </ImmersiveLayout>
  );
};

export default MetricsPage;
