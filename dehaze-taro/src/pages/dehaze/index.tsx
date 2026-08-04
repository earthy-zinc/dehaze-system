/**
 * 去雾 Tab 根页面（占位）
 *
 * 规划（05）：完整处理流程 —— 上传 → 算法 → 参数 → 处理 → 对比（页内步骤流）
 */
import React from "react";
import { View, Text } from "@tarojs/components";
import { PhotoOutlined } from "@taroify/icons";
import PageLayout from "@/layout";
import "./index.less";

const steps = ["上传图像", "选择算法", "调节参数", "处理", "效果对比"];

const DehazePage: React.FC = () => (
  <PageLayout level="L1" title="去雾">
    <View className="dehaze-page">
      {/* 步骤条（规划：页内步骤流） */}
      <View className="steps">
        {steps.map((step, i) => (
          <View key={step} className="step">
            <View className={`step-dot ${i === 0 ? "done" : ""}`}>{i + 1}</View>
            <Text className="step-label">{step}</Text>
          </View>
        ))}
      </View>

      {/* 流程内容占位 */}
      <View className="flow-placeholder">
        <PhotoOutlined size="44" color="#d1d5db" />
        <Text className="placeholder-text">去雾处理流程建设中</Text>
        <Text className="placeholder-sub">
          上传 → 选择算法 → 调节参数 → 处理 → 对比
        </Text>
      </View>

      <View className="dev-tip">
        <Text>核心流程按《菜单与页面层级规划》逐步落地</Text>
      </View>
    </View>
  </PageLayout>
);

export default DehazePage;
