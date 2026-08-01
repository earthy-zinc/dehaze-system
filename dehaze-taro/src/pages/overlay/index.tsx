import React, { useState, useEffect, useRef, useCallback } from "react";
import { View, Text, Image, ScrollView } from "@tarojs/components";
import { Button } from "@taroify/core";
import type { BaseEventOrig } from "@tarojs/components";
import Taro from "@tarojs/taro";
import { ModelAPI } from "dehaze-sdk-js";
import CompareNavbar from "@/components/compare/CompareNavbar";
import CompareToolbar from "@/components/compare/CompareToolbar";
import AlgorithmInfoCard from "@/components/compare/AlgorithmInfoCard";
import { loadCompareContext } from "@/components/compare/types";
import EmptyState from "@/components/common/EmptyState";
import "./index.less";

// Taro 的 BaseEventOrig 类型定义不完整（缺 touches），扩展为触摸事件类型
type TaroTouchEvent = BaseEventOrig & {
  touches: Array<{ clientX: number; clientY: number }>;
};

// PredEvalTaskStatus: 1 = PENDING, 2 = COMPLETED, 3 = FAILED
type TaskStatus = 1 | 2 | 3;

const OverlayPage: React.FC = () => {
  const [ctx] = useState(loadCompareContext);
  const [sliderPos, setSliderPos] = useState(50);
  const [reportLoading, setReportLoading] = useState(false);
  const [reportDownloading, setReportDownloading] = useState(false);
  // 缓存容器边界信息（跨端兼容：小程序不支持 getBoundingClientRect）
  const containerRectRef = useRef<{
    left: number;
    top: number;
    width: number;
    height: number;
  } | null>(null);
  const isDragging = useRef(false);

  const { originImage, result, algorithm } = ctx;
  const hasResult = originImage && result?.resultUrl;

  // 生成并下载报告
  const handleExportReport = async () => {
    if (!algorithm?.id || !originImage?.cleanUrl || !result?.resultUrl) {
      Taro.showToast({ title: "缺少必要参数，无法生成报告", icon: "none" });
      return;
    }

    setReportLoading(true);
    try {
      const res = await ModelAPI.generateReport({
        logId: algorithm.id,
        format: "pdf",
        includeMetrics: true,
        includeFilters: true,
      });
      const taskId = res.taskId;
      if (!taskId) {
        throw new Error("未返回任务ID");
      }

      // 轮询状态
      while (true) {
        const statusRes = await ModelAPI.getReportStatus(taskId);
        const status = statusRes.status as TaskStatus;
        if (status === 2) {
          // 已完成
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
        if (status === 3) {
          // 失败
          throw new Error(statusRes.errorMessage || "报告生成失败");
        }
        // status === 1，等待后重试
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

  // 查询容器尺寸（使用 Taro 节点查询 API，兼容小程序）
  useEffect(() => {
    if (!hasResult) return;
    const timer = setTimeout(() => {
      const query = Taro.createSelectorQuery();
      query.select(".image-container").boundingClientRect();
      query.exec((res) => {
        if (res && res[0]) {
          containerRectRef.current = res[0];
        }
      });
    }, 300);
    return () => clearTimeout(timer);
  }, [hasResult]);

  // 触摸开始
  const handleTouchStart = useCallback((e: BaseEventOrig) => {
    isDragging.current = true;
    const touch = (e as TaroTouchEvent).touches[0];
    const rect = containerRectRef.current;
    if (rect) {
      const x = touch.clientX - rect.left;
      const pos = (x / rect.width) * 100;
      setSliderPos(Math.max(0, Math.min(100, pos)));
    }
  }, []);

  // 触摸移动
  const handleTouchMove = useCallback((e: BaseEventOrig) => {
    if (!isDragging.current) return;
    const touch = (e as TaroTouchEvent).touches[0];
    const rect = containerRectRef.current;
    if (rect) {
      const x = touch.clientX - rect.left;
      const pos = (x / rect.width) * 100;
      setSliderPos(Math.max(0, Math.min(100, pos)));
    }
  }, []);

  // 触摸结束
  const handleTouchEnd = useCallback(() => {
    isDragging.current = false;
  }, []);

  return (
    <View className="overlay-page">
      {/* 顶部导航 */}
      <CompareNavbar title="重叠对比" />

      {/* 重叠对比区域 */}
      <ScrollView className="overlay-content" scrollY>
        {!hasResult ? (
          <EmptyState type="compare" />
        ) : (
          <>
            <View
              className="image-container"
              onTouchStart={handleTouchStart}
              onTouchMove={handleTouchMove}
              onTouchEnd={handleTouchEnd}
            >
              {/* 底层：处理后图片 */}
              <Image
                src={result!.resultUrl || ""}
                className="base-image"
                mode="widthFix"
                lazyLoad
              />
              {/* 上层：原图，通过 clip-path 控制显示区域 */}
              <View
                className="overlay-image-wrapper"
                style={{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }}
              >
                <Image
                  src={originImage!.url || ""}
                  className="overlay-image"
                  mode="widthFix"
                  lazyLoad
                />
              </View>
              {/* 滑动分隔线 */}
              <View
                className="slider-divider"
                style={{ left: `${sliderPos}%` }}
              >
                <View className="slider-line" />
                <View className="slider-handle">
                  <Text className="slider-arrow-left">◀</Text>
                  <Text className="slider-arrow-right">▶</Text>
                </View>
              </View>
              {/* 标签 */}
              <View className="image-labels">
                <View
                  className="label-tag label-original"
                  style={{ opacity: sliderPos > 10 ? 1 : 0 }}
                >
                  <Text>原图</Text>
                </View>
                <View
                  className="label-tag label-result"
                  style={{ opacity: sliderPos < 90 ? 1 : 0 }}
                >
                  <Text>处理后</Text>
                </View>
              </View>
            </View>

            <View className="overlay-hint">
              <Text>← 拖动分隔线对比效果 →</Text>
            </View>

            {/* 算法信息 */}
            <AlgorithmInfoCard algorithm={algorithm} result={result} />

            {/* 导出报告按钮 */}
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
          </>
        )}
      </ScrollView>

      {/* 底部工具栏 */}
      <CompareToolbar currentMode="overlay" resultUrl={result?.resultUrl} />
    </View>
  );
};

export default OverlayPage;
