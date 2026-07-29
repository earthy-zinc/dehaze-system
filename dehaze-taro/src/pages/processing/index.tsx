import React, { useState, useEffect, useCallback, useRef } from "react";
import { View, Text, Image, Slider } from "@tarojs/components";
import Taro from "@tarojs/taro";
import CompareNavbar from "@/components/compare/CompareNavbar";
import { ModelAPI } from "dehaze-sdk-js";
import type { Algorithm, PredictionResultVO } from "dehaze-sdk-js";
import { uploadImage } from "@/utils/upload";
import { formatFileSize, formatDuration } from "@/utils/format";
import { getErrorMessage } from "@/utils/error";
import { saveImageToAlbum } from "@/utils/saveImage";
import "./index.less";

interface ImageData {
  url: string;
  name: string;
  width: number;
  height: number;
  size: number;
}

type ProcessStatus = "idle" | "processing" | "success" | "error";

// 通用处理参数
interface ProcessParams {
  strength: number; // 去雾强度 0-100
  saturation: number; // 饱和度 0-200
  contrast: number; // 对比度 0-200
  sharpen: number; // 锐化 0-100
}

const DEFAULT_PARAMS: ProcessParams = {
  strength: 50,
  saturation: 100,
  contrast: 100,
  sharpen: 30,
};

// 根据算法类型预估处理时间（秒）
function estimateTime(algorithm: Algorithm | null): number {
  if (!algorithm) return 5;
  const type = (algorithm.type || "").toLowerCase();
  if (
    type.includes("cnn") ||
    type.includes("gan") ||
    type.includes("transformer") ||
    type.includes("深度")
  ) {
    return 6;
  }
  if (
    type.includes("传统") ||
    type.includes("dcp") ||
    type.includes("retinex")
  ) {
    return 2;
  }
  return 4;
}

const ProcessingPage: React.FC = () => {
  const [currentImage, setCurrentImage] = useState<ImageData | null>(null);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<Algorithm | null>(
    null
  );
  const [status, setStatus] = useState<ProcessStatus>("idle");
  const [result, setResult] = useState<PredictionResultVO | null>(null);
  const [errorMsg, setErrorMsg] = useState("");
  const [params, setParams] = useState<ProcessParams>(DEFAULT_PARAMS);
  const [showParams, setShowParams] = useState(false);
  const [elapsedTime, setElapsedTime] = useState(0);

  const elapsedTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // 缓存已上传的文件 ID，重试时复用，避免重复上传
  const uploadedFileIdRef = useRef<number | null>(null);

  // 清理定时器
  const clearAllTimers = useCallback(() => {
    if (elapsedTimerRef.current) {
      clearInterval(elapsedTimerRef.current);
      elapsedTimerRef.current = null;
    }
  }, []);

  useEffect(() => {
    return clearAllTimers;
  }, [clearAllTimers]);

  // 加载当前图片和算法
  useEffect(() => {
    try {
      const imgStr = Taro.getStorageSync("current_image");
      if (imgStr) {
        setCurrentImage(JSON.parse(imgStr));
      } else {
        Taro.showToast({ title: "请先选择图片", icon: "none" });
        setTimeout(() => Taro.navigateBack(), 1500);
        return;
      }
    } catch {
      Taro.showToast({ title: "图片数据读取失败", icon: "none" });
    }

    try {
      const algoStr = Taro.getStorageSync("selected_algorithm");
      if (algoStr) {
        setSelectedAlgorithm(JSON.parse(algoStr));
      } else {
        Taro.showToast({ title: "请先选择算法", icon: "none" });
        setTimeout(() => Taro.navigateBack(), 1500);
      }
    } catch {
      Taro.showToast({ title: "算法数据读取失败", icon: "none" });
    }
  }, []);

  // 启动已用时间计时（真实计时，非模拟进度）
  const startElapsedTimer = useCallback(() => {
    setElapsedTime(0);
    elapsedTimerRef.current = setInterval(() => {
      setElapsedTime((prev) => prev + 100);
    }, 100);
  }, []);

  // 实际执行处理
  const handleProcess = useCallback(async () => {
    if (!currentImage || !selectedAlgorithm) return;

    setStatus("processing");
    setErrorMsg("");
    setResult(null);
    startElapsedTimer();

    try {
      // 本地临时路径（blob:/wxfile://）服务端不可访问，需先上传换取 fileId
      if (!uploadedFileIdRef.current) {
        const fileInfo = await uploadImage(currentImage.url, currentImage.name);
        uploadedFileIdRef.current = fileInfo.id;
      }

      const res = await ModelAPI.predictAndWait({
        algorithmId: selectedAlgorithm.id,
        fileId: uploadedFileIdRef.current,
        params: JSON.stringify(params),
      });

      clearAllTimers();

      if (res.status === 3) {
        throw new Error(res.errorMessage || "处理失败");
      }

      setResult(res);
      setStatus("success");

      // 保存结果到 storage 供对比页面使用
      Taro.setStorageSync("prediction_result", JSON.stringify(res));

      Taro.showToast({ title: "处理完成", icon: "success" });

      // 处理完成后自动跳转对比页面（设计文档：处理完成 → 自动进入对比页面）
      setTimeout(() => {
        Taro.navigateTo({ url: "/pages/side-by-side/index" });
      }, 1500);
    } catch (error: unknown) {
      clearAllTimers();
      setStatus("error");
      setErrorMsg(getErrorMessage(error, "处理失败，请重试"));
      Taro.showToast({
        title: getErrorMessage(error, "处理失败"),
        icon: "none",
      });
    }
  }, [
    currentImage,
    selectedAlgorithm,
    params,
    startElapsedTimer,
    clearAllTimers,
  ]);

  // 执行去雾处理（带确认对话框）
  const handleStartProcess = useCallback(() => {
    if (!currentImage || !selectedAlgorithm) return;

    const estimatedSec = estimateTime(selectedAlgorithm);
    // 显示确认对话框（设计文档 2.1.3）
    Taro.showModal({
      title: "确认开始去雾处理",
      content: `图片：${currentImage.name}\n尺寸：${currentImage.width}×${currentImage.height}\n算法：${selectedAlgorithm.name}\n预估耗时：约 ${estimatedSec} 秒`,
      confirmText: "开始处理",
      cancelText: "取消",
      success: (res) => {
        if (res.confirm) {
          handleProcess();
        }
      },
    });
  }, [currentImage, selectedAlgorithm, handleProcess]);

  // 重新处理
  const handleRetry = useCallback(() => {
    setStatus("idle");
    setResult(null);
    setErrorMsg("");
    setElapsedTime(0);
  }, []);

  // 返回算法选择
  const handleBackToAlgorithm = useCallback(() => {
    Taro.navigateBack();
  }, []);

  // 保存结果到相册（设计文档 2.4.3）
  const handleSaveToAlbum = useCallback(async () => {
    if (!result?.resultUrl) return;
    await saveImageToAlbum(result.resultUrl);
  }, [result]);

  // 重置参数
  const handleResetParams = useCallback(() => {
    setParams(DEFAULT_PARAMS);
  }, []);

  const estimatedTime = estimateTime(selectedAlgorithm);

  return (
    <View className="processing-page">
      {/* 顶部导航 */}
      <CompareNavbar title="去雾处理" />

      <View className="processing-content">
        {/* 图片信息 */}
        {currentImage && (
          <View className="info-section">
            <Text className="section-label">原始图片</Text>
            <View className="image-preview">
              <Image
                src={currentImage.url}
                className="preview-img"
                mode="aspectFit"
                lazyLoad
              />
            </View>
            <View className="image-meta">
              <Text className="meta-item">{currentImage.name}</Text>
              <Text className="meta-item">
                {currentImage.width}×{currentImage.height}
              </Text>
              <Text className="meta-item">
                {formatFileSize(currentImage.size)}
              </Text>
            </View>
          </View>
        )}

        {/* 算法信息 */}
        {selectedAlgorithm && (
          <View className="info-section">
            <Text className="section-label">使用算法</Text>
            <View className="algorithm-info">
              <View className="algo-name-row">
                <Text className="algo-name">{selectedAlgorithm.name}</Text>
                <View className="algo-type-tag">
                  <Text>{selectedAlgorithm.type || "算法"}</Text>
                </View>
              </View>
              {selectedAlgorithm.description && (
                <Text className="algo-desc">
                  {selectedAlgorithm.description}
                </Text>
              )}
              <View className="algo-meta-row">
                {selectedAlgorithm.version && (
                  <Text className="algo-version">
                    版本: {selectedAlgorithm.version}
                  </Text>
                )}
                <Text className="algo-estimate">
                  预估耗时: 约 {estimatedTime} 秒
                </Text>
              </View>
            </View>
          </View>
        )}

        {/* 参数调节面板（idle 状态可用）*/}
        {status === "idle" && (
          <View className="info-section params-section">
            <View
              className="params-header"
              onClick={() => setShowParams((prev) => !prev)}
            >
              <Text className="section-label" style={{ marginBottom: 0 }}>
                参数调节
              </Text>
              <Text className="toggle-text">
                {showParams ? "收起" : "展开"}
              </Text>
            </View>

            {showParams && (
              <View className="params-body">
                <View className="param-item">
                  <View className="param-row">
                    <Text className="param-label">去雾强度</Text>
                    <Text className="param-value">{params.strength}</Text>
                  </View>
                  <Slider
                    min={0}
                    max={100}
                    value={params.strength}
                    onChanging={(e) =>
                      setParams((prev) => ({
                        ...prev,
                        strength: e.detail.value,
                      }))
                    }
                  />
                </View>

                <View className="param-item">
                  <View className="param-row">
                    <Text className="param-label">色彩饱和度</Text>
                    <Text className="param-value">{params.saturation}</Text>
                  </View>
                  <Slider
                    min={0}
                    max={200}
                    value={params.saturation}
                    onChanging={(e) =>
                      setParams((prev) => ({
                        ...prev,
                        saturation: e.detail.value,
                      }))
                    }
                  />
                </View>

                <View className="param-item">
                  <View className="param-row">
                    <Text className="param-label">对比度</Text>
                    <Text className="param-value">{params.contrast}</Text>
                  </View>
                  <Slider
                    min={0}
                    max={200}
                    value={params.contrast}
                    onChanging={(e) =>
                      setParams((prev) => ({
                        ...prev,
                        contrast: e.detail.value,
                      }))
                    }
                  />
                </View>

                <View className="param-item">
                  <View className="param-row">
                    <Text className="param-label">锐化程度</Text>
                    <Text className="param-value">{params.sharpen}</Text>
                  </View>
                  <Slider
                    min={0}
                    max={100}
                    value={params.sharpen}
                    onChanging={(e) =>
                      setParams((prev) => ({
                        ...prev,
                        sharpen: e.detail.value,
                      }))
                    }
                  />
                </View>

                <View className="params-actions">
                  <View className="reset-btn" onClick={handleResetParams}>
                    <Text>恢复默认</Text>
                  </View>
                </View>
              </View>
            )}
          </View>
        )}

        {/* 处理中：不定式加载（API 同步返回，不显示虚假百分比）*/}
        {status === "processing" && (
          <View className="status-section processing">
            <View className="processing-spinner" />
            <Text className="status-text">正在去雾处理中...</Text>
            <Text className="status-hint">
              已用 {formatDuration(elapsedTime)}
            </Text>
          </View>
        )}

        {/* 处理成功 */}
        {status === "success" && result && (
          <View className="status-section success">
            <View className="success-icon">
              <Text>✓</Text>
            </View>
            <Text className="status-text">处理完成</Text>
            <Text className="status-hint">
              耗时 {formatDuration(result.time ?? 0)}
              {result.fromCache ? " · 缓存命中" : ""}
            </Text>

            {/* 结果预览 */}
            {result.resultUrl && (
              <View className="result-preview">
                <Text className="section-label">处理结果</Text>
                <View className="result-image-wrapper">
                  <Image
                    src={result.resultUrl}
                    className="result-img"
                    mode="aspectFit"
                    lazyLoad
                  />
                </View>
              </View>
            )}

            <View className="action-buttons">
              <View
                className="btn btn-primary"
                onClick={() =>
                  Taro.navigateTo({ url: "/pages/side-by-side/index" })
                }
              >
                <Text>效果对比</Text>
              </View>
              <View className="btn btn-secondary" onClick={handleSaveToAlbum}>
                <Text>保存到相册</Text>
              </View>
            </View>
            <View className="action-buttons">
              <View className="btn btn-secondary" onClick={handleRetry}>
                <Text>重新处理</Text>
              </View>
            </View>
          </View>
        )}

        {/* 处理失败 */}
        {status === "error" && (
          <View className="status-section error">
            <View className="error-icon">
              <Text>!</Text>
            </View>
            <Text className="status-text">处理失败</Text>
            <Text className="status-hint">{errorMsg}</Text>
            <View className="action-buttons">
              <View className="btn btn-primary" onClick={handleProcess}>
                <Text>重试</Text>
              </View>
              <View
                className="btn btn-secondary"
                onClick={handleBackToAlgorithm}
              >
                <Text>更换算法</Text>
              </View>
            </View>
          </View>
        )}

        {/* 开始处理按钮 */}
        {status === "idle" && (
          <View className="action-section">
            <View
              className={`start-btn ${!currentImage || !selectedAlgorithm ? "disabled" : ""}`}
              onClick={handleStartProcess}
            >
              <Text>开始去雾</Text>
            </View>
            <View className="back-link" onClick={handleBackToAlgorithm}>
              <Text>← 重新选择算法</Text>
            </View>
          </View>
        )}
      </View>
    </View>
  );
};

export default ProcessingPage;
