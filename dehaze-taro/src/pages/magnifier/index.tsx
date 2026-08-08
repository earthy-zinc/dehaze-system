import React, { useState, useEffect, useRef, useCallback } from "react";
import { View, Text, Image } from "@tarojs/components";
import type { BaseEventOrig } from "@tarojs/components";
import Taro from "@tarojs/taro";
import { Button } from "@taroify/core";
import { ModelAPI } from "dehaze-sdk-js";
import ImmersiveLayout from "@/layout/immersive";
import CompareToolbar from "@/components/compare/CompareToolbar";
import { loadCompareContext } from "@/components/compare/types";
import EmptyState from "@/components/common/EmptyState";
import "./index.less";

// 放大倍数选项
const ZOOM_OPTIONS = [2, 3, 5] as const;
type ZoomValue = (typeof ZOOM_OPTIONS)[number];
// 放大镜尺寸选项
const SIZE_OPTIONS = [
  { value: 100, label: "小" },
  { value: 150, label: "中" },
  { value: 200, label: "大" },
] as const;
// 显示模式
type DisplayMode = "origin" | "result" | "compare";

// Taro 的 BaseEventOrig 类型定义不完整（缺 touches），扩展为触摸事件类型
type TaroTouchEvent = BaseEventOrig & {
  touches: Array<{ clientX: number; clientY: number }>;
};

// PredEvalTaskStatus: 1 = PENDING, 2 = COMPLETED, 3 = FAILED
type TaskStatus = 1 | 2 | 3;

const MagnifierPage: React.FC = () => {
  const [ctx] = useState(loadCompareContext);
  const [zoom, setZoom] = useState<ZoomValue>(2);
  const [lensSize, setLensSize] = useState<number>(150);
  const [displayMode, setDisplayMode] = useState<DisplayMode>("compare");
  const [lensPos, setLensPos] = useState({ x: 0, y: 0 });
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });
  const [reportLoading, setReportLoading] = useState(false);
  const [reportDownloading, setReportDownloading] = useState(false);

  // 缓存容器边界信息（跨端兼容：小程序不支持 getBoundingClientRect）
  const containerRectRef = useRef<{
    left: number;
    top: number;
    width: number;
    height: number;
  } | null>(null);

  const { originImage, result } = ctx;
  const hasResult = originImage && result?.resultUrl;

  // 获取容器尺寸（使用 Taro 节点查询 API，兼容小程序）
  useEffect(() => {
    if (!hasResult) return;
    const timer = setTimeout(() => {
      const query = Taro.createSelectorQuery();
      query.select(".image-container").boundingClientRect();
      query.exec((res) => {
        if (res && res[0]) {
          const rect = res[0];
          containerRectRef.current = rect;
          setContainerSize({ width: rect.width, height: rect.height });
          setLensPos({ x: rect.width / 2, y: rect.height / 2 });
        }
      });
    }, 300);
    return () => clearTimeout(timer);
  }, [hasResult]);

  // 触摸移动放大镜
  const handleTouchMove = useCallback((e: BaseEventOrig) => {
    const { touches } = e as TaroTouchEvent;
    const touch = touches[0];
    const rect = containerRectRef.current;
    if (!rect) return;
    const x = touch.clientX - rect.left;
    const y = touch.clientY - rect.top;
    setLensPos({
      x: Math.max(0, Math.min(rect.width, x)),
      y: Math.max(0, Math.min(rect.height, y)),
    });
  }, []);

  // 双指捏合调整倍数
  const lastPinchDistance = useRef(0);
  const handleTouchStart = useCallback((e: BaseEventOrig) => {
    const { touches } = e as TaroTouchEvent;
    if (touches.length === 2) {
      const dx = touches[0].clientX - touches[1].clientX;
      const dy = touches[0].clientY - touches[1].clientY;
      lastPinchDistance.current = Math.sqrt(dx * dx + dy * dy);
    }
  }, []);

  const handlePinchMove = useCallback((e: BaseEventOrig) => {
    const { touches } = e as TaroTouchEvent;
    if (touches.length !== 2 || lastPinchDistance.current === 0) return;
    const dx = touches[0].clientX - touches[1].clientX;
    const dy = touches[0].clientY - touches[1].clientY;
    const distance = Math.sqrt(dx * dx + dy * dy);
    const delta = distance - lastPinchDistance.current;

    if (Math.abs(delta) > 10) {
      setZoom((prev) => {
        const currentIndex = ZOOM_OPTIONS.indexOf(prev);
        if (delta > 0) {
          const nextIndex = Math.min(currentIndex + 1, ZOOM_OPTIONS.length - 1);
          return ZOOM_OPTIONS[nextIndex];
        } else {
          const nextIndex = Math.max(currentIndex - 1, 0);
          return ZOOM_OPTIONS[nextIndex];
        }
      });
      lastPinchDistance.current = distance;
    }
  }, []);

  const handleTouchEnd = useCallback(() => {
    lastPinchDistance.current = 0;
  }, []);

  // 生成并下载报告
  const handleExportReport = async () => {
    if (!result?.resultUrl) {
      Taro.showToast({ title: "缺少必要参数，无法生成报告", icon: "none" });
      return;
    }
    setReportLoading(true);
    try {
      const res = await ModelAPI.generateReport({ logId: 0, format: "pdf" });
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
              const filePath = await Taro.downloadFile({ url: statusRes.downloadUrl });
              if (filePath.tempFilePath) {
                await Taro.openDocument({ filePath: filePath.tempFilePath, showMenu: true });
              }
            } catch { Taro.showToast({ title: "打开报告失败", icon: "none" }); }
            finally { setReportDownloading(false); }
          } else { throw new Error("报告生成但无下载链接"); }
          break;
        }
        if (status === 3) throw new Error(statusRes.errorMessage || "报告生成失败");
        await new Promise((r) => setTimeout(r, 2000));
      }
    } catch (err: unknown) {
      Taro.showToast({ title: err instanceof Error ? err.message : "报告生成失败", icon: "none" });
    } finally { setReportLoading(false); }
  };

  // 点击切换显示模式（原图 → 处理后 → 对比 → 原图）
  const handleTap = useCallback(() => {
    setDisplayMode((prev) => {
      if (prev === "origin") return "result";
      if (prev === "result") return "compare";
      return "origin";
    });
  }, []);

  // 计算放大镜内背景图位置
  const getLensBackgroundPosition = () => {
    if (!containerSize.width || !containerSize.height) return "0 0";
    const bgX = -(lensPos.x * zoom - lensSize / 2);
    const bgY = -(lensPos.y * zoom - lensSize / 2);
    return `${bgX}px ${bgY}px`;
  };

  const lensStyle = (imageUrl: string): React.CSSProperties => ({
    width: `${lensSize}px`,
    height: `${lensSize}px`,
    left: `${lensPos.x - lensSize / 2}px`,
    top: `${lensPos.y - lensSize / 2}px`,
    backgroundImage: `url(${imageUrl})`,
    backgroundRepeat: "no-repeat",
    backgroundSize: `${containerSize.width * zoom}px ${containerSize.height * zoom}px`,
    backgroundPosition: getLensBackgroundPosition(),
    borderRadius: "50%",
    border: "4rpx solid #fff",
    boxShadow: "0 4rpx 16rpx rgb(0 0 0 / 30%)",
  });

  return (
    <ImmersiveLayout
      title="放大镜对比"
      toolbar={
        <CompareToolbar currentMode="magnifier" resultUrl={result?.resultUrl} resultId={result?.logId} />
      }
    >
      {/* 对比区域 */}
      {!hasResult ? (
        <EmptyState type="compare" />
      ) : (
        <>
          {/* 图片容器 + 放大镜 */}
          <View
            className="image-container"
            onTouchStart={handleTouchStart}
            onTouchMove={(e) => {
              handleTouchMove(e);
              handlePinchMove(e);
            }}
            onTouchEnd={handleTouchEnd}
            onClick={handleTap}
          >
            <Image
              src={result!.resultUrl || ""}
              className="base-image"
              mode="widthFix"
              lazyLoad
            />

            {/* 原图放大镜 */}
            {(displayMode === "origin" || displayMode === "compare") && (
              <View
                className={`magnifier-lens ${displayMode === "compare" ? "lens-origin" : ""}`}
                style={lensStyle(originImage!.url)}
              />
            )}

            {/* 处理后放大镜 */}
            {(displayMode === "result" || displayMode === "compare") && (
              <View
                className={`magnifier-lens ${displayMode === "compare" ? "lens-result" : ""}`}
                style={{
                  ...lensStyle(result!.resultUrl || ""),
                  ...(displayMode === "compare"
                    ? { left: `${lensPos.x + lensSize / 2}px` }
                    : {}),
                }}
              />
            )}
          </View>

          {/* 提示 */}
          <View className="magnifier-hint">
            <Text>拖动移动放大镜 · 双指捏合调整倍数 · 点击切换模式</Text>
          </View>

          {/* 控制面板 */}
          <View className="control-panel">
            {/* 显示模式 */}
            <View className="control-group">
              <Text className="control-label">显示模式</Text>
              <View className="control-options">
                {(
                  [
                    { value: "origin", label: "原图" },
                    { value: "result", label: "处理后" },
                    { value: "compare", label: "对比" },
                  ] as const
                ).map((opt) => (
                  <View
                    key={opt.value}
                    className={`control-option ${displayMode === opt.value ? "active" : ""}`}
                    onClick={() => setDisplayMode(opt.value)}
                  >
                    <Text>{opt.label}</Text>
                  </View>
                ))}
              </View>
            </View>

            {/* 放大倍数 */}
            <View className="control-group">
              <Text className="control-label">放大倍数</Text>
              <View className="control-options">
                {ZOOM_OPTIONS.map((z) => (
                  <View
                    key={z}
                    className={`control-option ${zoom === z ? "active" : ""}`}
                    onClick={() => setZoom(z)}
                  >
                    <Text>{z}x</Text>
                  </View>
                ))}
              </View>
            </View>

            {/* 放大镜尺寸 */}
            <View className="control-group">
              <Text className="control-label">放大镜大小</Text>
              <View className="control-options">
                {SIZE_OPTIONS.map((opt) => (
                  <View
                    key={opt.value}
                    className={`control-option ${lensSize === opt.value ? "active" : ""}`}
                    onClick={() => setLensSize(opt.value)}
                  >
                    <Text>{opt.label}</Text>
                  </View>
                ))}
              </View>
            </View>
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
        </>
      )}
    </ImmersiveLayout>
  );
};

export default MagnifierPage;
