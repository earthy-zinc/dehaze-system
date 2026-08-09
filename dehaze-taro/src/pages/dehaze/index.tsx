/**
 * 去雾 Tab 根页面（重构为页内步骤流）
 *
 * 按 05 规划 2.3：上传 → 算法选择 → 参数调节 → 处理 → 对比（页内步骤流）
 */
import React, { useState, useEffect, useCallback, useRef } from "react";
import { View, Text, Image, Button, Slider, Input, ScrollView } from "@tarojs/components";
import Taro from "@tarojs/taro";
import { PhotoOutlined, Search, Arrow } from "@taroify/icons";
import PageLayout from "@/layout";
import { AlgorithmAPI, ModelAPI } from "dehaze-sdk-js";
import type { Algorithm, PredictionResultVO, PresetVO } from "dehaze-sdk-js";
import { uploadImage } from "@/config/upload";
import { getErrorMessage } from "@/utils/error";
import { formatFileSize, formatDuration } from "@/utils/format";
import { useProcessStore } from "@/stores/process";
import "./index.less";

type StepKey = "upload" | "algorithm" | "params" | "processing" | "compare";

interface StepDef {
  key: StepKey;
  label: string;
}

const STEPS: StepDef[] = [
  { key: "upload", label: "上传图像" },
  { key: "algorithm", label: "选择算法" },
  { key: "params", label: "调节参数" },
  { key: "processing", label: "处理" },
  { key: "compare", label: "效果对比" },
];

interface ImageData {
  url: string;
  name: string;
  width: number;
  height: number;
  size: number;
}

interface ProcessParams {
  strength: number;
  saturation: number;
  contrast: number;
  sharpen: number;
}

const DEFAULT_PARAMS: ProcessParams = {
  strength: 50,
  saturation: 100,
  contrast: 100,
  sharpen: 30,
};

type ProcessStatus = "idle" | "processing" | "success" | "error";

/** 重试间隔（毫秒）：2s → 5s → 10s */
const RETRY_DELAYS = [2000, 5000, 10000];
const MAX_RETRIES = 3;

const DehazePage: React.FC = () => {
  const [currentStep, setCurrentStep] = useState(0);
  const [currentImage, setCurrentImage] = useState<ImageData | null>(null);
  const [algorithms, setAlgorithms] = useState<Algorithm[]>([]);
  const [algoLoading, setAlgoLoading] = useState(false);
  const [algoSearch, setAlgoSearch] = useState("");
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<Algorithm | null>(
    null
  );
  const [params, setParams] = useState<ProcessParams>(DEFAULT_PARAMS);
  const [processStatus, setProcessStatus] = useState<ProcessStatus>("idle");
  const [result, setResult] = useState<PredictionResultVO | null>(null);
  const [errorMsg, setErrorMsg] = useState("");
  const [elapsedTime, setElapsedTime] = useState(0);
  const [presets, setPresets] = useState<PresetVO[]>([]);
  const uploadedFileIdRef = useRef<number | null>(null);
  const elapsedTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const retryCountRef = useRef(0);
  const cancelledRef = useRef(false);

  const clearTimer = useCallback(() => {
    if (elapsedTimerRef.current) {
      clearInterval(elapsedTimerRef.current);
      elapsedTimerRef.current = null;
    }
  }, []);

  useEffect(() => {
    return clearTimer;
  }, [clearTimer]);

  // 切换到算法选择步骤时加载算法
  useEffect(() => {
    if (currentStep === 1 && algorithms.length === 0 && !algoLoading) {
      setAlgoLoading(true);
      AlgorithmAPI.getList()
        .then((data) => setAlgorithms(data || []))
        .catch(() => Taro.showToast({ title: "加载算法失败", icon: "none" }))
        .finally(() => setAlgoLoading(false));
    }
  }, [currentStep, algorithms.length, algoLoading]);

  // 加载参数预设
  useEffect(() => {
    ModelAPI.getPresets({ pageNum: 1, pageSize: 50 })
      .then((res) => setPresets(res.list || []))
      .catch(() => { /* 静默 */ });
  }, []);

  // 步骤跳转
  const goStep = useCallback(
    (step: number) => {
      if (step < 0 || step >= STEPS.length) return;
      // 不允许跳过前面的步骤
      if (step > currentStep) {
        if (step >= 1 && !currentImage) {
          Taro.showToast({ title: "请先上传图片", icon: "none" });
          return;
        }
        if (step >= 2 && !selectedAlgorithm) {
          Taro.showToast({ title: "请先选择算法", icon: "none" });
          return;
        }
      }
      setCurrentStep(step);
    },
    [currentStep, currentImage, selectedAlgorithm]
  );

  // 上传图片
  const handleChooseImage = useCallback(async () => {
    try {
      const res = await Taro.chooseMedia({
        count: 1,
        mediaType: ["image"],
        sourceType: ["album", "camera"],
        sizeType: ["original", "compressed"],
      });
      const file = res.tempFiles[0];
      const info = await Taro.getImageInfo({ src: file.tempFilePath });
      setCurrentImage({
        url: file.tempFilePath,
        name: `图片_${Date.now()}`,
        width: info.width,
        height: info.height,
        size: file.size,
      });
      Taro.showToast({ title: "上传成功", icon: "success" });
    } catch (error: unknown) {
      const errMsg = (error as { errMsg?: string })?.errMsg || "";
      if (!errMsg.includes("cancel")) {
        Taro.showToast({ title: "选择图片失败", icon: "none" });
      }
    }
  }, []);

  // 收集所有叶子算法
  const collectLeaves = useCallback((nodes: Algorithm[]): Algorithm[] => {
    const leaves: Algorithm[] = [];
    const walk = (list: Algorithm[]) => {
      for (const node of list) {
        if (node.children && node.children.length > 0) {
          walk(node.children);
        } else {
          leaves.push(node);
        }
      }
    };
    walk(nodes);
    return leaves;
  }, []);

  const leafAlgorithms = collectLeaves(algorithms).filter(
    (a) => a.status === 4
  );

  const filteredAlgorithms = algoSearch
    ? leafAlgorithms.filter(
        (a) =>
          a.name?.toLowerCase().includes(algoSearch.toLowerCase()) ||
          a.description?.toLowerCase().includes(algoSearch.toLowerCase())
      )
    : leafAlgorithms;

  // 选择算法
  const handleSelectAlgorithm = useCallback(
    (algorithm: Algorithm) => {
      setSelectedAlgorithm(algorithm);
      goStep(2);
    },
    [goStep]
  );

  // 应用预设
  const handleApplyPreset = useCallback((preset: PresetVO) => {
    try {
      const presetParams = JSON.parse(preset.params);
      setParams({
        strength: presetParams.strength ?? DEFAULT_PARAMS.strength,
        saturation: presetParams.saturation ?? DEFAULT_PARAMS.saturation,
        contrast: presetParams.contrast ?? DEFAULT_PARAMS.contrast,
        sharpen: presetParams.sharpen ?? DEFAULT_PARAMS.sharpen,
      });
    } catch {
      Taro.showToast({ title: "预设参数解析失败", icon: "none" });
    }
  }, []);

  // 执行处理（含递增重试）
  const handleProcess = useCallback(async () => {
    if (!currentImage || !selectedAlgorithm) return;

    // 配额检查
    try {
      const quota = await ModelAPI.getQuota();
      if (quota.remaining === 0) {
        Taro.showModal({
          title: "预测次数不足",
          content: `当前剩余预测次数为 0（已使用 ${quota.used}/${quota.total}），请及时充值。`,
          confirmText: "去充值",
          cancelText: "取消",
          success: (res) => {
            if (res.confirm) {
              Taro.navigateTo({ url: "/pages/user-center/index" });
            }
          },
        });
        return;
      }
    } catch {
      // 配额查询失败也允许继续处理
    }

    // 确认对话框
    const confirmResult = await new Promise<boolean>((resolve) => {
      Taro.showModal({
        title: "确认开始去雾处理",
        content: `图片：${currentImage.name}\n尺寸：${currentImage.width}×${currentImage.height}\n算法：${selectedAlgorithm.name}`,
        confirmText: "开始处理",
        cancelText: "取消",
        success: (res) => resolve(res.confirm),
        fail: () => resolve(false),
      });
    });
    if (!confirmResult) return;

    setProcessStatus("processing");
    setErrorMsg("");
    setResult(null);
    cancelledRef.current = false;
    elapsedTimerRef.current = setInterval(() => {
      setElapsedTime((prev) => prev + 100);
    }, 100);

    const attempt = async (attemptNumber: number): Promise<void> => {
      if (cancelledRef.current) return;

      try {
        if (!uploadedFileIdRef.current) {
          const fileInfo = await uploadImage(currentImage.url, currentImage.name);
          uploadedFileIdRef.current = fileInfo.id;
        }

        const res = await ModelAPI.predictAndWait({
          algorithmId: selectedAlgorithm.id,
          fileId: uploadedFileIdRef.current,
          params: JSON.stringify(params),
        });

        if (cancelledRef.current) return;
        clearTimer();

        if (res.status === 3) {
          throw new Error(res.errorMessage || "处理失败");
        }

        setResult(res);
        setProcessStatus("success");
        retryCountRef.current = 0;
        useProcessStore.getState().setResult(res);
        Taro.showToast({ title: "处理完成", icon: "success" });
        goStep(4);
      } catch (error: unknown) {
        if (cancelledRef.current) return;

        const errMsg = getErrorMessage(error, "处理失败");
        if (attemptNumber < MAX_RETRIES) {
          const delay = RETRY_DELAYS[attemptNumber];
          retryCountRef.current = attemptNumber + 1;
          setErrorMsg(`${errMsg}，${delay / 1000}秒后自动重试（${attemptNumber + 1}/${MAX_RETRIES}）`);
          await new Promise((r) => setTimeout(r, delay));
          if (!cancelledRef.current) {
            return attempt(attemptNumber + 1);
          }
        }

        clearTimer();
        setProcessStatus("error");
        setErrorMsg(errMsg);
      }
    };

    await attempt(0);
  }, [currentImage, selectedAlgorithm, params, clearTimer, goStep]);

  // 进入对比页
  const handleGoCompare = useCallback(() => {
    Taro.navigateTo({ url: "/pages/side-by-side/index" });
  }, []);

  // 取消处理
  const handleCancelProcess = useCallback(() => {
    cancelledRef.current = true;
    clearTimer();
    setProcessStatus("idle");
    setErrorMsg("");
    setElapsedTime(0);
    Taro.showToast({ title: "已取消处理", icon: "none" });
  }, [clearTimer]);

  // 重置
  const handleReset = useCallback(() => {
    setCurrentStep(0);
    setCurrentImage(null);
    setSelectedAlgorithm(null);
    setParams(DEFAULT_PARAMS);
    setProcessStatus("idle");
    setResult(null);
    setErrorMsg("");
    setElapsedTime(0);
    uploadedFileIdRef.current = null;
    retryCountRef.current = 0;
    clearTimer();
  }, [clearTimer]);

  // 渲染步骤指示器
  const renderSteps = () => (
    <View className="dehaze-steps">
      {STEPS.map((step, i) => {
        const done = i < currentStep;
        const active = i === currentStep;
        return (
          <View key={step.key} className="dehaze-step">
            <View
              className={`dehaze-step-dot ${done ? "done" : ""} ${active ? "active" : ""}`}
            >
              {done ? <Text>✓</Text> : <Text>{i + 1}</Text>}
            </View>
            <Text
              className={`dehaze-step-label ${active ? "active-label" : ""}`}
            >
              {step.label}
            </Text>
          </View>
        );
      })}
    </View>
  );

  // 步骤1：上传
  const renderUpload = () => (
    <View className="dehaze-step-content">
      <View className="dehaze-upload-area" onClick={handleChooseImage}>
        {currentImage ? (
          <Image
            src={currentImage.url}
            className="dehaze-upload-preview"
            mode="aspectFit"
            lazyLoad
          />
        ) : (
          <View className="dehaze-upload-placeholder">
            <PhotoOutlined size="48" color="#9ca3af" />
            <Text className="dehaze-upload-text">点击选择图片</Text>
            <Text className="dehaze-upload-hint">支持相册或拍照</Text>
          </View>
        )}
      </View>
      {currentImage && (
        <View className="dehaze-image-info">
          <Text className="dehaze-info-item">{currentImage.name}</Text>
          <Text className="dehaze-info-item">
            {currentImage.width}×{currentImage.height}
          </Text>
          <Text className="dehaze-info-item">
            {formatFileSize(currentImage.size)}
          </Text>
        </View>
      )}
      <View className="dehaze-step-action">
        <Button
          className={`dehaze-btn dehaze-btn-primary ${!currentImage ? "disabled" : ""}`}
          onClick={() => goStep(1)}
          disabled={!currentImage}
        >
          下一步：选择算法
          <Arrow size="14" color="#fff" />
        </Button>
      </View>
    </View>
  );

  // 步骤2：选择算法
  const renderAlgorithmSelect = () => (
    <View className="dehaze-step-content">
      <View className="dehaze-algo-search">
        <Search size="16" color="#9ca3af" />
        <Input
          className="dehaze-algo-search-input"
          placeholder="搜索算法..."
          value={algoSearch}
          onInput={(e) => setAlgoSearch(e.detail.value)}
        />
      </View>
      {algoLoading ? (
        <View className="dehaze-loading">
          <Text>加载算法中...</Text>
        </View>
      ) : (
        <View className="dehaze-algo-list">
          {filteredAlgorithms.map((algo) => (
            <View
              key={algo.id}
              className={`dehaze-algo-card ${selectedAlgorithm?.id === algo.id ? "selected" : ""}`}
              onClick={() => handleSelectAlgorithm(algo)}
            >
              <Text className="dehaze-algo-name">{algo.name}</Text>
              {algo.type && (
                <View className="dehaze-algo-type">
                  <Text>{algo.type}</Text>
                </View>
              )}
              {algo.description && (
                <Text className="dehaze-algo-desc">{algo.description}</Text>
              )}
            </View>
          ))}
        </View>
      )}
      <View className="dehaze-step-action">
        <View
          className="dehaze-btn dehaze-btn-secondary"
          onClick={() => goStep(0)}
        >
          返回上一步
        </View>
        <Button
          className={`dehaze-btn dehaze-btn-primary ${!selectedAlgorithm ? "disabled" : ""}`}
          onClick={() => goStep(2)}
          disabled={!selectedAlgorithm}
        >
          下一步：调节参数
          <Arrow size="14" color="#fff" />
        </Button>
      </View>
    </View>
  );

  // 步骤3：参数调节
  const renderParams = () => (
    <View className="dehaze-step-content">
      <View className="dehaze-params-section">
        {/* 预设选择 */}
        {presets.length > 0 && (
          <View className="dehaze-preset-section">
            <Text className="dehaze-preset-label">参数预设</Text>
            <ScrollView scrollX className="dehaze-preset-scroll" enhanced showScrollbar={false}>
              {presets.map((preset) => (
                <View
                  key={preset.id}
                  className="dehaze-preset-chip"
                  onClick={() => handleApplyPreset(preset)}
                >
                  <Text>{preset.name}</Text>
                </View>
              ))}
            </ScrollView>
          </View>
        )}

        <View className="dehaze-param-item">
          <View className="dehaze-param-header">
            <Text className="dehaze-param-label">去雾强度</Text>
            <Text className="dehaze-param-value">{params.strength}</Text>
          </View>
          <Slider
            min={0}
            max={100}
            value={params.strength}
            onChanging={(e) =>
              setParams((prev) => ({ ...prev, strength: e.detail.value }))
            }
          />
        </View>
        <View className="dehaze-param-item">
          <View className="dehaze-param-header">
            <Text className="dehaze-param-label">色彩饱和度</Text>
            <Text className="dehaze-param-value">{params.saturation}</Text>
          </View>
          <Slider
            min={0}
            max={200}
            value={params.saturation}
            onChanging={(e) =>
              setParams((prev) => ({ ...prev, saturation: e.detail.value }))
            }
          />
        </View>
        <View className="dehaze-param-item">
          <View className="dehaze-param-header">
            <Text className="dehaze-param-label">对比度</Text>
            <Text className="dehaze-param-value">{params.contrast}</Text>
          </View>
          <Slider
            min={0}
            max={200}
            value={params.contrast}
            onChanging={(e) =>
              setParams((prev) => ({ ...prev, contrast: e.detail.value }))
            }
          />
        </View>
        <View className="dehaze-param-item">
          <View className="dehaze-param-header">
            <Text className="dehaze-param-label">锐化程度</Text>
            <Text className="dehaze-param-value">{params.sharpen}</Text>
          </View>
          <Slider
            min={0}
            max={100}
            value={params.sharpen}
            onChanging={(e) =>
              setParams((prev) => ({ ...prev, sharpen: e.detail.value }))
            }
          />
        </View>
        <View
          className="dehaze-param-reset"
          onClick={() => setParams(DEFAULT_PARAMS)}
        >
          <Text>恢复默认</Text>
        </View>
      </View>

      {selectedAlgorithm && (
        <View className="dehaze-algo-summary">
          <Text className="dehaze-summary-label">已选算法：</Text>
          <Text className="dehaze-summary-value">{selectedAlgorithm.name}</Text>
        </View>
      )}

      <View className="dehaze-step-action">
        <View
          className="dehaze-btn dehaze-btn-secondary"
          onClick={() => goStep(1)}
        >
          返回上一步
        </View>
        <Button
          className="dehaze-btn dehaze-btn-primary"
          onClick={handleProcess}
        >
          开始去雾
          <Arrow size="14" color="#fff" />
        </Button>
      </View>
    </View>
  );

  // 步骤4：处理中
  const renderProcessing = () => (
    <View className="dehaze-step-content">
      {processStatus === "processing" && (
        <View className="dehaze-processing-status">
          <View className="dehaze-spinner" />
          <Text className="dehaze-status-text">正在去雾处理中...</Text>
          <Text className="dehaze-status-hint">
            已用 {formatDuration(elapsedTime)}
          </Text>
          <View className="dehaze-step-action">
            <View
              className="dehaze-btn dehaze-btn-secondary"
              onClick={handleCancelProcess}
            >
              取消处理
            </View>
          </View>
        </View>
      )}
      {processStatus === "success" && result && (
        <View className="dehaze-success-status">
          <View className="dehaze-success-icon">✓</View>
          <Text className="dehaze-status-text">处理完成</Text>
          <Text className="dehaze-status-hint">
            耗时 {formatDuration(result.time ?? 0)}
            {result.fromCache ? " · 缓存命中" : ""}
          </Text>
          {result.resultUrl && (
            <View className="dehaze-result-preview">
              <Image
                src={result.resultUrl}
                className="dehaze-result-img"
                mode="aspectFit"
                lazyLoad
              />
            </View>
          )}
          <View className="dehaze-step-action">
            <Button
              className="dehaze-btn dehaze-btn-primary"
              onClick={handleGoCompare}
            >
              进入效果对比
              <Arrow size="14" color="#fff" />
            </Button>
            <View
              className="dehaze-btn dehaze-btn-secondary"
              onClick={handleReset}
            >
              重新开始
            </View>
          </View>
        </View>
      )}
      {processStatus === "error" && (
        <View className="dehaze-error-status">
          <View className="dehaze-error-icon">!</View>
          <Text className="dehaze-status-text">处理失败</Text>
          <Text className="dehaze-status-hint">{errorMsg}</Text>
          <View className="dehaze-step-action">
            <View
              className="dehaze-btn dehaze-btn-primary"
              onClick={handleProcess}
            >
              重试
            </View>
            <View
              className="dehaze-btn dehaze-btn-secondary"
              onClick={() => goStep(2)}
            >
              调整参数
            </View>
          </View>
        </View>
      )}
    </View>
  );

  // 步骤5：对比入口
  const renderCompare = () => (
    <View className="dehaze-step-content">
      <View className="dehaze-compare-entry">
        <View className="dehaze-compare-icon">
          <Text>⟷</Text>
        </View>
        <Text className="dehaze-compare-title">效果对比</Text>
        <Text className="dehaze-compare-desc">
          查看处理前后的对比效果，支持并排、重叠、放大镜等多种模式
        </Text>
        <View className="dehaze-step-action">
          <Button
            className="dehaze-btn dehaze-btn-primary"
            onClick={handleGoCompare}
          >
            进入效果对比
            <Arrow size="14" color="#fff" />
          </Button>
          <View
            className="dehaze-btn dehaze-btn-secondary"
            onClick={handleReset}
          >
            开始新的去雾
          </View>
        </View>
      </View>
    </View>
  );

  const stepRenderers: Record<StepKey, () => React.ReactNode> = {
    upload: renderUpload,
    algorithm: renderAlgorithmSelect,
    params: renderParams,
    processing: renderProcessing,
    compare: renderCompare,
  };

  return (
    <PageLayout level="L1" title="去雾">
      <View className="dehaze-page">
        {renderSteps()}
        <View className="dehaze-flow-body">
          {stepRenderers[STEPS[currentStep].key]()}
        </View>
      </View>
    </PageLayout>
  );
};

export default DehazePage;
