/**
 * 批量处理页面（L2）
 * 批量上传图片 → 选择模型 → 逐张提交 → 进度追踪 → 结果查看
 */
import React, { useState, useCallback } from "react";
import { View, Text, Image, ScrollView } from "@tarojs/components";
import Taro, { useLoad } from "@tarojs/taro";
import { Navbar, Button, Loading, Progress } from "@taroify/core";
import { ArrowLeft, Plus, Cross } from "@taroify/icons";
import { AiImageAPI, MemberAPI } from "clarity-sdk-js";
import type { AiModelVO } from "clarity-sdk-js";
import { getErrorMessage } from "@/utils/error";
import { uploadImage } from "@/config/upload";
import { loadImageEditModels } from "@/utils/imageModel";
import "./index.less";

interface BatchItem {
  id: string;
  localPath: string;
  fileId?: number;
  status: "pending" | "processing" | "completed" | "failed";
  resultUrl?: string;
  errorMessage?: string;
  time?: number;
  taskId?: number;
}

/** 会员未配置批量上限时的默认值 */
const DEFAULT_MAX_IMAGES = 5;
const RETRY_DELAYS = [2000, 5000, 10000];
const MAX_RETRIES = 3;

const BatchPage: React.FC = () => {
  const [images, setImages] = useState<BatchItem[]>([]);
  const [models, setModels] = useState<AiModelVO[]>([]);
  const [modelLoading, setModelLoading] = useState(true);
  const [selectedModelId, setSelectedModelId] = useState<string>("");
  const [params, setParams] = useState("");
  const [processing, setProcessing] = useState(false);
  const [maxImages, setMaxImages] = useState(DEFAULT_MAX_IMAGES);

  // 加载图像编辑模型 + 会员批量上限
  useLoad(() => {
    setModelLoading(true);
    loadImageEditModels()
      .then(({ models: list, preselected }) => {
        setModels(list);
        setSelectedModelId((prev) => prev || preselected?.modelId || "");
      })
      .catch(() => {
        Taro.showToast({ title: "加载模型失败", icon: "none" });
      })
      .finally(() => setModelLoading(false));

    MemberAPI.getProfile()
      .then((profile) => {
        setMaxImages(profile.benefits?.batchLimit || DEFAULT_MAX_IMAGES);
      })
      .catch(() => {
        /* 非会员使用默认上限 */
      });
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

  // 提交单张图片（本地临时路径服务端不可访问，需先上传换取 fileId）
  const processOne = useCallback(
    async (img: BatchItem) => {
      setImages((prev) =>
        prev.map((item) =>
          item.id === img.id ? { ...item, status: "processing" } : item
        )
      );

      const attempt = async (attemptNumber: number): Promise<void> => {
        try {
          const fileId =
            img.fileId ??
            (await uploadImage(img.localPath, `batch_${img.id}`)).id;
          const result = await AiImageAPI.submitAndWait({
            model: selectedModelId,
            fileId,
            params: params || undefined,
          });
          setImages((prev) =>
            prev.map((item) => {
              if (item.id !== img.id) return item;
              if (result.status === 2) {
                return {
                  ...item,
                  fileId,
                  status: "completed",
                  resultUrl: result.resultUrl,
                  time: result.time,
                  taskId: result.taskId,
                };
              }
              return {
                ...item,
                fileId,
                status: "failed",
                errorMessage: result.errorMessage,
              };
            })
          );
        } catch (err: unknown) {
          if (attemptNumber < MAX_RETRIES) {
            await new Promise((resolve) =>
              setTimeout(resolve, RETRY_DELAYS[attemptNumber])
            );
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
    [selectedModelId, params]
  );

  // 开始批量处理：批量端点已退役，按多张独立任务提交
  const handleStartBatch = useCallback(async () => {
    if (!selectedModelId) {
      Taro.showToast({ title: "请先选择模型", icon: "none" });
      return;
    }
    if (images.length === 0) {
      Taro.showToast({ title: "请先上传图片", icon: "none" });
      return;
    }

    setProcessing(true);
    setImages((prev) => prev.map((img) => ({ ...img, status: "pending" })));
    try {
      await Promise.all(images.map(processOne));
    } finally {
      setProcessing(false);
    }
  }, [selectedModelId, images, processOne]);

  // 重试单张
  const handleRetryImage = useCallback(
    (img: BatchItem) => {
      if (!selectedModelId) return;
      return processOne(img);
    },
    [selectedModelId, processOne]
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

        {/* 模型选择 */}
        <View className="section">
          <Text className="section-title">选择模型</Text>
          {modelLoading ? (
            <View className="algo-loading">
              <Loading size="16" />
              <Text>加载模型中...</Text>
            </View>
          ) : (
            <View className="algo-list">
              {models.map((model) => (
                <View
                  key={model.id}
                  className={`algo-card ${selectedModelId === model.modelId ? "selected" : ""}`}
                  onClick={() => setSelectedModelId(model.modelId)}
                >
                  <Text className="algo-name">{model.displayName}</Text>
                  <Text className="algo-desc">{model.modelId}</Text>
                  {selectedModelId === model.modelId && (
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
              onInput={(e) =>
                setParams((e.target as HTMLTextAreaElement).value)
              }
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
                              耗时 {img.time}s
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
            disabled={images.length === 0 || !selectedModelId || processing}
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
