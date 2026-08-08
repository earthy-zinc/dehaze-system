import React, { useState } from "react";
import { View, Text, Image, ScrollView } from "@tarojs/components";
import Taro from "@tarojs/taro";
import { Button } from "@taroify/core";
import { ModelAPI } from "dehaze-sdk-js";
import ImmersiveLayout from "@/layout/immersive";
import CompareToolbar from "@/components/compare/CompareToolbar";
import AlgorithmInfoCard from "@/components/compare/AlgorithmInfoCard";
import { loadCompareContext } from "@/components/compare/types";
import EmptyState from "@/components/common/EmptyState";
import "./index.less";

// PredEvalTaskStatus: 1 = PENDING, 2 = COMPLETED, 3 = FAILED
type TaskStatus = 1 | 2 | 3;

const SideBySidePage: React.FC = () => {
  const [ctx] = useState(loadCompareContext);
  const [reportLoading, setReportLoading] = useState(false);
  const [reportDownloading, setReportDownloading] = useState(false);

  const { originImage, result, algorithm } = ctx;
  const hasResult = originImage && result?.resultUrl;

  // 生成并下载报告
  const handleExportReport = async () => {
    if (!originImage?.cleanUrl || !result?.resultUrl) {
      Taro.showToast({ title: "缺少必要参数，无法生成报告", icon: "none" });
      return;
    }

    setReportLoading(true);
    try {
      const res = await ModelAPI.generateReport({
        logId: 0,
        format: "pdf",
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

  return (
    <ImmersiveLayout
      title="效果对比"
      toolbar={
        <CompareToolbar
          currentMode="side-by-side"
          resultUrl={result?.resultUrl}
          resultId={result?.logId}
        />
      }
    >
      <ScrollView className="compare-content" scrollY>
        {!hasResult ? (
          <EmptyState type="compare" />
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
                  src={originImage?.url || ""}
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
                  src={result?.resultUrl || ""}
                  className="compare-image"
                  mode="widthFix"
                  lazyLoad
                />
              </View>
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
    </ImmersiveLayout>
  );
};

export default SideBySidePage;
