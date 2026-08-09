/**
 * 批量处理页面（L2）
 * 批量上传图片 → 选择算法 → 批量执行 → 进度追踪 → 结果查看
 */
import React, { useState, useCallback } from "react";
import { View, Text, Image, ScrollView } from "@tarojs/components";
import Taro, { useLoad } from "@tarojs/taro";
import { Navbar, Button, Loading, Progress } from "@taroify/core";
import { ArrowLeft, Plus, Cross } from "@taroify/icons";
import { ModelAPI, AlgorithmAPI } from "dehaze-sdk-js";
import type { Algorithm } from "dehaze-sdk-js";
import { getErrorMessage } from "@/utils/error";
import "./index.less";

interface BatchItem {
  id: string;
  localPath: string;
  fileId?: number;
  status: "pending" | "processing" | "completed" | "failed";
  resultUrl?: string;
  errorMessage?: string;
  time?: number;
  logId?: number;
}

/** 会员等级对应的批量处理上限 */
const VIP_MAX_IMAGES: Record<string, number> = {
  svip: 20,
  VIP2: 15,
  VIP1: 10,
  default: 5,
};
const RETRY_DELAYS = [2000, 5000, 10000];
const MAX_RETRIES = 3;

const BatchPage: React.FC = () => {
  const [images, setImages] = useState<BatchItem[]>([]);
  const [algorithms, setAlgorithms] = useState<Algorithm[]>([]);
  const [algoLoading, setAlgoLoading] = useState(true);
  const [selectedAlgoId, setSelectedAlgoId] = useState<number | null>(null);
  const [params, setParams] = useState("");
  const [processing, setProcessing] = useState(false);
  const [maxImages, setMaxImages] = useState(20);

  // 加载算法列表 + 获取会员上限
  useLoad(() => {
    setAlgoLoading(true);
    AlgorithmAPI.getList()
      .then((data) => setAlgorithms(data || []))
      .catch(() => {
        Taro.showToast({ title: "加载算法失败", icon: "none" });
      })
      .finally(() => setAlgoLoading(false));

    // 获取配额信息来推断会员等级上限
    ModelAPI.getQuota()
      .then((quota) => {
        const total = quota.total || 0;
        if (total >= 200) setMaxImages(VIP_MAX_IMAGES.svip);
        else if (total >= 150) setMaxImages(VIP_MAX_IMAGES.VIP2);
        else if (total >= 100) setMaxImages(VIP_MAX_IMAGES.VIP1);
        else setMaxImages(VIP_MAX_IMAGES.default);
      })
      .catch(() => { /* 使用默认上限 */ });
  });

  // 选择图片
  const handleChooseImages = useCallback(() => {
    const remain = maxImages - images.length;
    if (remain <= 0) {
      Taro.showToast({ title: `最多${maxImages}张图片`, icon: "none" });
      return;
    }
    Taro.chooseImage({
      count: remain,
      sizeType: ["compressed"],
      sourceType: ["album", "camera"],
      success: (res) => {
        const newImages: BatchItem[] = res.tempFilePaths.map((path, idx) => ({
          id: `${Date.now()}_${idx}`,
          localPath: path,
          status: "pending",
        }));
        setImages((prev) => [...prev, ...newImages]);
      },
    });
  }, [images.length, maxImages]);

  // 移除图片
  const removeImage = useCallback((id: string) => {
    setImages((prev) => prev.filter((img) => img.id !== id));
  }, []);

  // 开始批量处理
  const handleStartBatch = useCallback(async () => {
    if (!selectedAlgoId) {
      Taro.showToast({ title: "请先选择算法", icon: "none" });
      return;
    }
    if (images.length === 0) {
      Taro.showToast({ title: "请先上传图片", icon: "none" });
      return;
    }

    setProcessing(true);
    setImages((prev) => prev.map((img) => ({ ...img, status: "pending" })));

    try {
      const result = await ModelAPI.batchPredict({
        algorithmId: selectedAlgoId,
        items: images.map((img) => ({
          imageUrl: img.localPath,
          params: params || undefined,
        })),
      });

      // 批量预测返回多结果
      if (result.results && result.results.length > 0) {
        setImages((prev) =>
          prev.map((img, idx) => {
            const r = result.results[idx];
            if (!r) return img;
            if (r.status === 2) {
              return {
                ...img,
                status: "completed",
                resultUrl: r.resultUrl,
                time: r.time,
              };
            }
            if (r.status === 3) {
              return { ...img, status: "failed", errorMessage: r.errorMessage };
            }
            return { ...img, status: "processing", logId: r.logId };
          })
        );

        // 轮询未完成的任务
        const pendingTasks = result.results
          .map((r, idx) => ({ ...r, idx }))
          .filter((r) => r.status === 1);

        if (pendingTasks.length > 0) {
          const pollPromises = pendingTasks.map(async (task) => {
            try {
              const finalResult = await ModelAPI.predictAndWait(
                {
                  algorithmId: selectedAlgoId,
                  imageUrl: images[task.idx].localPath,
                  params: params || undefined,
                },
                { intervalMs: 2000, timeoutMs: 120000 }
              );
              setImages((prev) =>
                prev.map((img, idx) => {
                  if (idx !== task.idx) return img;
                  if (finalResult.status === 2) {
                    return {
                      ...img,
                      status: "completed",
                      resultUrl: finalResult.resultUrl,
                      time: finalResult.time,
                    };
                  }
                  return {
                    ...img,
                    status: "failed",
                    errorMessage: finalResult.errorMessage,
                  };
                })
              );
            } catch {
              setImages((prev) =>
                prev.map((img, idx) =>
                  idx === task.idx
                    ? { ...img, status: "failed", errorMessage: "处理超时" }
                    : img
                )
              );
            }
          });
          await Promise.all(pollPromises);
        }
      }
    } catch (err: unknown) {
      Taro.showToast({
        title: getErrorMessage(err, "批量处理失败"),
        icon: "none",
      });
      setImages((prev) =>
        prev.map((img) =>
          img.status === "pending"
            ? { ...img, status: "failed", errorMessage: "提交失败" }
            : img
        )
      );
    } finally {
      setProcessing(false);
    }
  }, [selectedAlgoId, images, params]);

  // 重试单张（含递增重试间隔）
  const handleRetryImage = useCallback(
    async (img: BatchItem) => {
      if (!selectedAlgoId) return;
      setImages((prev) =>
        prev.map((item) =>
          item.id === img.id ? { ...item, status: "processing" } : item
        )
      );

      const attempt = async (attemptNumber: number): Promise<void> => {
        try {
          const result = await ModelAPI.predictAndWait({
            algorithmId: selectedAlgoId,
            imageUrl: img.localPath,
            params: params || undefined,
          });
          setImages((prev) =>
            prev.map((item) => {
              if (item.id !== img.id) return item;
              if (result.status === 2) {
                return {
                  ...item,
                  status: "completed",
                  resultUrl: result.resultUrl,
                  time: result.time,
                };
              }
              return {
                ...item,
                status: "failed",
                errorMessage: result.errorMessage,
              };
            })
          );
        } catch (err: unknown) {
          if (attemptNumber < MAX_RETRIES) {
            await new Promise((r) => setTimeout(r, RETRY_DELAYS[attemptNumber]));
            return attempt(attemptNumber + 1);
          }
          setImages((prev) =>
            prev.map((item) =>
              item.id === img.id
                ? {
                    ...item,
                    status: "failed",
                    errorMessage: getErrorMessage(err, "处理失败"),
                  }
                : item
            )
          );
        }
      };

      await attempt(0);
    },
    [selectedAlgoId, params]
  );

  // 预览结果图
  const handlePreview = useCallback((url: string) => {
    Taro.previewImage({ urls: [url], current: url });
  }, []);

  const completedCount = images.filter((i) => i.status === "completed").length;
  const failedCount = images.filter((i) => i.status === "failed").length;
  const totalProgress =
    images.length > 0
      ? Math.round(((completedCount + failedCount) / images.length) * 100)
      : 0;

  return (
    <View className="batch-page">
      <Navbar title="批量处理">
        <Navbar.NavLeft>
          <ArrowLeft />
        </Navbar.NavLeft>
      </Navbar>

      <ScrollView className="batch-content" scrollY>
        {/* 上传区域 */}
        <View className="section">
          <Text className="section-title">选择图片</Text>
          <Text className="section-hint">
            最多{maxImages}张，已选{images.length}张
          </Text>

          <View className="image-grid">
            {images.map((img) => (
              <View key={img.id} className="image-item">
                <Image
                  src={img.localPath}
                  className="image-thumb"
                  mode="aspectFill"
                />
                <View
                  className="image-remove"
                  onClick={() => removeImage(img.id)}
                >
                  <Cross size="12" color="#fff" />
                </View>
                {img.status === "processing" && (
                  <View className="image-status processing">
                    <Loading size="16" />
                  </View>
                )}
                {img.status === "completed" && (
                  <View className="image-status completed">
                    <Text>✓</Text>
                  </View>
                )}
                {img.status === "failed" && (
                  <View className="image-status failed">
                    <Text>!</Text>
                  </View>
                )}
              </View>
            ))}

            {images.length < maxImages && (
              <View className="image-add" onClick={handleChooseImages}>
                <Plus size="32" color="#3b82f6" />
                <Text className="add-text">添加图片</Text>
              </View>
            )}
          </View>
        </View>

        {/* 算法选择 */}
        <View className="section">
          <Text className="section-title">选择算法</Text>
          {algoLoading ? (
            <View className="algo-loading">
              <Loading size="16" />
              <Text>加载算法中...</Text>
            </View>
          ) : (
            <View className="algo-list">
              {algorithms
                .filter((a) => a.status === 4 && !a.children?.length)
                .map((algo) => (
                  <View
                    key={algo.id}
                    className={`algo-card ${selectedAlgoId === algo.id ? "selected" : ""}`}
                    onClick={() => setSelectedAlgoId(algo.id)}
                  >
                    <Text className="algo-name">{algo.name}</Text>
                    {algo.description && (
                      <Text className="algo-desc">{algo.description}</Text>
                    )}
                    {selectedAlgoId === algo.id && (
                      <View className="algo-check">✓</View>
                    )}
                  </View>
                ))}
            </View>
          )}
        </View>

        {/* 参数 */}
        <View className="section">
          <Text className="section-title">参数（可选）</Text>
          <View className="params-input-wrapper">
            <textarea
              className="params-input"
              placeholder='JSON参数，如 {"strength":0.8}'
              value={params}
              onInput={(e) => setParams((e.target as HTMLTextAreaElement).value)}
            />
          </View>
        </View>

        {/* 进度 */}
        {processing && (
          <View className="section">
            <Text className="section-title">处理进度</Text>
            <View className="progress-card">
              <Progress percent={totalProgress} />
              <Text className="progress-text">
                已完成 {completedCount} / 失败 {failedCount} / 总计{" "}
                {images.length}
              </Text>
            </View>
          </View>
        )}

        {/* 结果列表 */}
        {!processing && (completedCount > 0 || failedCount > 0) && (
          <View className="section">
            <Text className="section-title">处理结果</Text>
            <View className="result-list">
              {images
                .filter(
                  (i) => i.status === "completed" || i.status === "failed"
                )
                .map((img) => (
                  <View key={img.id} className={`result-item ${img.status}`}>
                    <Image
                      src={img.localPath}
                      className="result-thumb"
                      mode="aspectFill"
                    />
                    <View className="result-info">
                      {img.status === "completed" ? (
                        <>
                          <Text className="result-status success-text">
                            处理完成
                          </Text>
                          {img.time && (
                            <Text className="result-time">
                              耗时 {img.time}ms
                            </Text>
                          )}
                          <View className="result-actions">
                            <Button
                              size="mini"
                              onClick={() => handlePreview(img.resultUrl!)}
                            >
                              查看结果
                            </Button>
                          </View>
                        </>
                      ) : (
                        <>
                          <Text className="result-status error-text">
                            处理失败
                          </Text>
                          {img.errorMessage && (
                            <Text className="result-error">
                              {img.errorMessage}
                            </Text>
                          )}
                          <Button
                            size="mini"
                            color="primary"
                            onClick={() => handleRetryImage(img)}
                          >
                            重试
                          </Button>
                        </>
                      )}
                    </View>
                  </View>
                ))}
            </View>
          </View>
        )}

        {/* 底部操作 */}
        <View className="bottom-bar">
          <Button
            block
            color="primary"
            loading={processing}
            disabled={images.length === 0 || !selectedAlgoId || processing}
            onClick={handleStartBatch}
          >
            {processing ? "处理中..." : "开始批量处理"}
          </Button>
        </View>
      </ScrollView>
    </View>
  );
};

export default BatchPage;
