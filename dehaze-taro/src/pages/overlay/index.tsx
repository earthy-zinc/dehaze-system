import React, { useState, useEffect, useRef, useCallback } from "react";
import { View, Text, Image, ScrollView } from "@tarojs/components";
import Taro from "@tarojs/taro";
import { ArrowLeft } from "@taroify/icons";
import CompareToolbar from "@/components/compare/CompareToolbar";
import AlgorithmInfoCard from "@/components/compare/AlgorithmInfoCard";
import { loadCompareContext } from "@/components/compare/types";
import "./index.less";

const OverlayPage: React.FC = () => {
  const [ctx, setCtx] = useState(loadCompareContext);
  const [sliderPos, setSliderPos] = useState(50);
  const containerRef = useRef<HTMLDivElement>(null);
  const isDragging = useRef(false);

  useEffect(() => {
    setCtx(loadCompareContext());
  }, []);

  const { originImage, result, algorithm } = ctx;

  // 触摸开始
  const handleTouchStart = useCallback((e: any) => {
    isDragging.current = true;
    const touch = e.touches[0];
    const rect = containerRef.current?.getBoundingClientRect();
    if (rect) {
      const x = touch.clientX - rect.left;
      const pos = (x / rect.width) * 100;
      setSliderPos(Math.max(0, Math.min(100, pos)));
    }
  }, []);

  // 触摸移动
  const handleTouchMove = useCallback((e: any) => {
    if (!isDragging.current) return;
    const touch = e.touches[0];
    const rect = containerRef.current?.getBoundingClientRect();
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

  const hasResult = originImage && result?.resultUrl;

  return (
    <View className="overlay-page">
      {/* 顶部导航 */}
      <View className="navbar">
        <View className="nav-back" onClick={() => Taro.navigateBack()}>
          <ArrowLeft size="20" color="#333" />
        </View>
        <Text className="nav-title">重叠对比</Text>
      </View>

      {/* 重叠对比区域 */}
      <ScrollView className="overlay-content" scrollY>
        {!hasResult ? (
          <View className="empty-state">
            <Text className="empty-text">暂无对比数据</Text>
            <Text className="empty-hint">请先完成去雾处理</Text>
          </View>
        ) : (
          <>
            <View
              className="image-container"
              ref={containerRef as any}
              onTouchStart={handleTouchStart}
              onTouchMove={handleTouchMove}
              onTouchEnd={handleTouchEnd}
            >
              {/* 底层：处理后图片 */}
              <Image
                src={result!.resultUrl}
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
                  src={originImage!.url}
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
          </>
        )}
      </ScrollView>

      {/* 底部工具栏 */}
      <CompareToolbar currentMode="overlay" resultUrl={result?.resultUrl} />
    </View>
  );
};

export default OverlayPage;
