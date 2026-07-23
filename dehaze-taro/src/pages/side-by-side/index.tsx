import React, { useState } from "react";
import { View, Text, Image, ScrollView } from "@tarojs/components";
import CompareNavbar from "@/components/compare/CompareNavbar";
import CompareToolbar from "@/components/compare/CompareToolbar";
import AlgorithmInfoCard from "@/components/compare/AlgorithmInfoCard";
import { loadCompareContext } from "@/components/compare/types";
import "./index.less";

const SideBySidePage: React.FC = () => {
  const [ctx] = useState(loadCompareContext);

  const { originImage, result, algorithm } = ctx;
  const hasResult = originImage && result?.resultUrl;

  return (
    <View className="side-by-side-page">
      {/* 顶部导航 */}
      <CompareNavbar title="效果对比" />

      {/* 对比内容 */}
      <ScrollView className="compare-content" scrollY>
        {!hasResult ? (
          <View className="empty-state">
            <Text className="empty-text">暂无对比数据</Text>
            <Text className="empty-hint">请先完成去雾处理</Text>
          </View>
        ) : (
          <>
            {/* 原图 */}
            <View className="image-section">
              <View className="image-label">
                <View className="label-tag label-original">
                  <Text>原图</Text>
                </View>
                <Text className="image-name">{originImage!.name}</Text>
              </View>
              <View className="image-wrapper">
                <Image
                  src={originImage!.url}
                  className="compare-image"
                  mode="widthFix"
                  lazyLoad
                />
              </View>
            </View>

            {/* 分隔线 */}
            <View className="image-divider">
              <View className="divider-line" />
            </View>

            {/* 处理后 */}
            <View className="image-section">
              <View className="image-label">
                <View className="label-tag label-result">
                  <Text>处理后</Text>
                </View>
                <Text className="image-name">
                  {algorithm?.name || "去雾结果"}
                </Text>
              </View>
              <View className="image-wrapper">
                <Image
                  src={result!.resultUrl}
                  className="compare-image"
                  mode="widthFix"
                  lazyLoad
                />
              </View>
            </View>

            {/* 算法信息 */}
            <AlgorithmInfoCard algorithm={algorithm} result={result} />
          </>
        )}
      </ScrollView>

      {/* 底部工具栏 */}
      <CompareToolbar
        currentMode="side-by-side"
        resultUrl={result?.resultUrl}
      />
    </View>
  );
};

export default SideBySidePage;
