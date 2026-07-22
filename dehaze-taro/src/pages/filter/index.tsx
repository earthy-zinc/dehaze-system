import React, { useState, useEffect, useMemo, useCallback, useRef } from "react";
import { View, Text, Image, ScrollView, Slider, Canvas } from "@tarojs/components";
import Taro from "@tarojs/taro";
import { ArrowLeft } from "@taroify/icons";
import CompareToolbar from "@/components/compare/CompareToolbar";
import { loadCompareContext } from "@/components/compare/types";
import "./index.less";

// 滤镜参数类型
interface FilterParams {
  brightness: number; // 亮度 -100~100
  contrast: number; // 对比度 -100~100
  saturation: number; // 饱和度 -100~100
  temperature: number; // 色温 -100~100
  sharpen: number; // 锐化 0~100
  denoise: number; // 降噪 0~100
}

// 默认参数
const DEFAULT_PARAMS: FilterParams = {
  brightness: 0,
  contrast: 0,
  saturation: 0,
  temperature: 0,
  sharpen: 0,
  denoise: 0,
};

// 滤镜配置
const FILTER_CONFIGS: {
  key: keyof FilterParams;
  label: string;
  min: number;
  max: number;
}[] = [
  { key: "brightness", label: "亮度", min: -100, max: 100 },
  { key: "contrast", label: "对比度", min: -100, max: 100 },
  { key: "saturation", label: "饱和度", min: -100, max: 100 },
  { key: "temperature", label: "色温", min: -100, max: 100 },
  { key: "sharpen", label: "锐化", min: 0, max: 100 },
  { key: "denoise", label: "降噪", min: 0, max: 100 },
];

// 内置预设
const BUILTIN_PRESETS: { name: string; params: FilterParams }[] = [
  {
    name: "自然",
    params: {
      brightness: 5,
      contrast: 10,
      saturation: 5,
      temperature: 0,
      sharpen: 0,
      denoise: 0,
    },
  },
  {
    name: "鲜艳",
    params: {
      brightness: 0,
      contrast: 30,
      saturation: 40,
      temperature: 0,
      sharpen: 0,
      denoise: 0,
    },
  },
  {
    name: "柔和",
    params: {
      brightness: 0,
      contrast: -20,
      saturation: 0,
      temperature: 0,
      sharpen: -10,
      denoise: 0,
    },
  },
  {
    name: "清晰",
    params: {
      brightness: 0,
      contrast: 20,
      saturation: 0,
      temperature: 0,
      sharpen: 40,
      denoise: 0,
    },
  },
  {
    name: "复古",
    params: {
      brightness: 0,
      contrast: 0,
      saturation: -20,
      temperature: 30,
      sharpen: 0,
      denoise: 0,
    },
  },
];

// 自定义预设 storage key
const CUSTOM_PRESETS_KEY = "custom_filter_presets";

const FilterPage: React.FC = () => {
  const [ctx, setCtx] = useState(loadCompareContext);
  const [params, setParams] = useState<FilterParams>(DEFAULT_PARAMS);
  const [showOrigin, setShowOrigin] = useState(false);
  const [customPresets, setCustomPresets] = useState<
    { name: string; params: FilterParams }[]
  >([]);

  // 小程序端 Canvas 相关状态
  const [canvasDisplaySize, setCanvasDisplaySize] = useState({
    width: 0,
    height: 0,
  });
  const canvasNodeRef = useRef<any>(null);
  const canvasCtxRef = useRef<any>(null);

  useEffect(() => {
    setCtx(loadCompareContext());
    // 加载自定义预设
    try {
      const stored = Taro.getStorageSync(CUSTOM_PRESETS_KEY);
      if (stored) setCustomPresets(JSON.parse(stored));
    } catch {
      /* ignore */
    }
  }, []);

  const { result } = ctx;
  const hasResult = result?.resultUrl;

  // 构建 CSS filter 字符串（H5 端 Image style 与小程序端 Canvas ctx.filter 共用）
  const filterStyle = useMemo(() => {
    // brightness: 1 + value/100
    const brightness = 1 + params.brightness / 100;
    // contrast: 1 + value/100
    const contrast = 1 + params.contrast / 100;
    // saturation: 1 + value/100
    const saturation = 1 + params.saturation / 100;
    // 色温: 用 sepia + hue-rotate 近似
    const sepia = Math.abs(params.temperature) / 100;
    const hueRotate = params.temperature * 0.5;
    // 锐化/降噪: CSS 不直接支持，用 contrast 微调近似
    const sharpenBoost = 1 + params.sharpen / 200;
    const denoiseBlur = params.denoise / 200;

    return `brightness(${brightness}) contrast(${contrast * sharpenBoost}) saturate(${saturation}) sepia(${sepia}) hue-rotate(${hueRotate}deg) blur(${denoiseBlur}px)`;
  }, [params]);

  // 当前预览图 URL
  const currentPreviewUrl = useMemo(() => {
    if (showOrigin) return ctx.originImage?.url || "";
    return result?.resultUrl || "";
  }, [showOrigin, ctx.originImage, result]);

  // 小程序端：获取图片尺寸以确定 Canvas 显示尺寸
  useEffect(() => {
    if (process.env.TARO_ENV === "h5") return;
    if (!hasResult || !currentPreviewUrl) {
      setCanvasDisplaySize({ width: 0, height: 0 });
      return;
    }

    let cancelled = false;
    Taro.getImageInfo({ src: currentPreviewUrl })
      .then((info) => {
        if (cancelled) return;
        const sysInfo = Taro.getSystemInfoSync();
        const maxWidth = sysInfo.windowWidth;
        const maxHeight = sysInfo.windowHeight * 0.4; // 对应 40vh
        let displayWidth = maxWidth;
        let displayHeight = (info.height / info.width) * displayWidth;
        if (displayHeight > maxHeight) {
          displayHeight = maxHeight;
          displayWidth = (info.width / info.height) * displayHeight;
        }
        setCanvasDisplaySize({ width: displayWidth, height: displayHeight });
      })
      .catch(() => {
        if (cancelled) return;
        const sysInfo = Taro.getSystemInfoSync();
        setCanvasDisplaySize({
          width: sysInfo.windowWidth,
          height: sysInfo.windowWidth * 0.75,
        });
      });

    return () => {
      cancelled = true;
    };
  }, [hasResult, currentPreviewUrl]);

  // 小程序端：在 Canvas 上绘制带滤镜的图片
  useEffect(() => {
    if (process.env.TARO_ENV === "h5") return;
    if (!hasResult || !currentPreviewUrl) return;
    if (canvasDisplaySize.width === 0 || canvasDisplaySize.height === 0) return;

    const query = Taro.createSelectorQuery();
    query
      .select("#filterCanvas")
      .fields({ node: true, size: true })
      .exec((res) => {
        if (!res || !res[0] || !res[0].node) return;
        const canvas = res[0].node;
        const ctx2d = canvas.getContext("2d") as CanvasRenderingContext2D;
        const dpr = Taro.getSystemInfoSync().pixelRatio;
        canvas.width = canvasDisplaySize.width * dpr;
        canvas.height = canvasDisplaySize.height * dpr;
        ctx2d.scale(dpr, dpr);

        canvasNodeRef.current = canvas;
        canvasCtxRef.current = ctx2d;

        const img = canvas.createImage();
        img.onload = () => {
          ctx2d.clearRect(
            0,
            0,
            canvasDisplaySize.width,
            canvasDisplaySize.height
          );
          // 应用滤镜（原图不应用滤镜）
          ctx2d.filter = showOrigin ? "none" : filterStyle;
          // aspectFit 绘制
          const imgAspect = img.width / img.height;
          const canvasAspect =
            canvasDisplaySize.width / canvasDisplaySize.height;
          let drawW: number;
          let drawH: number;
          let drawX: number;
          let drawY: number;
          if (imgAspect > canvasAspect) {
            drawW = canvasDisplaySize.width;
            drawH = drawW / imgAspect;
            drawX = 0;
            drawY = (canvasDisplaySize.height - drawH) / 2;
          } else {
            drawH = canvasDisplaySize.height;
            drawW = drawH * imgAspect;
            drawX = (canvasDisplaySize.width - drawW) / 2;
            drawY = 0;
          }
          ctx2d.drawImage(img, drawX, drawY, drawW, drawH);
        };
        img.src = currentPreviewUrl;
      });
  }, [canvasDisplaySize, filterStyle, showOrigin, hasResult, currentPreviewUrl]);

  // 参数变更
  const handleParamChange = useCallback(
    (key: keyof FilterParams, value: number) => {
      setParams((prev) => ({ ...prev, [key]: value }));
    },
    []
  );

  // 重置
  const handleReset = useCallback(() => {
    setParams(DEFAULT_PARAMS);
  }, []);

  // 应用预设
  const applyPreset = useCallback((presetParams: FilterParams) => {
    setParams(presetParams);
  }, []);

  // 保存自定义预设
  const handleSavePreset = useCallback(() => {
    Taro.showModal({
      title: "保存预设",
      editable: true,
      placeholderText: "请输入预设名称",
      success: (res: any) => {
        const name = (res.content || "").trim();
        if (res.confirm && name) {
          const newPreset = { name, params };
          const updated = [...customPresets, newPreset];
          setCustomPresets(updated);
          Taro.setStorageSync(CUSTOM_PRESETS_KEY, JSON.stringify(updated));
          Taro.showToast({ title: "预设已保存", icon: "success" });
        }
      },
    } as any);
  }, [params, customPresets]);

  // 删除自定义预设
  const handleDeletePreset = useCallback(
    (index: number) => {
      Taro.showModal({
        title: "确认删除",
        content: "确定要删除此自定义预设吗？",
        success: (res) => {
          if (res.confirm) {
            const updated = customPresets.filter((_, i) => i !== index);
            setCustomPresets(updated);
            Taro.setStorageSync(CUSTOM_PRESETS_KEY, JSON.stringify(updated));
            Taro.showToast({ title: "已删除", icon: "success" });
          }
        },
      });
    },
    [customPresets]
  );

  // 是否有参数变更
  const hasChanges = useMemo(() => {
    return (Object.keys(params) as (keyof FilterParams)[]).some(
      (key) => params[key] !== DEFAULT_PARAMS[key]
    );
  }, [params]);

  // 是否为 H5 环境
  const isH5 = process.env.TARO_ENV === "h5";

  return (
    <View className="filter-page">
      {/* 顶部导航 */}
      <View className="navbar">
        <View className="nav-back" onClick={() => Taro.navigateBack()}>
          <ArrowLeft size="20" color="#333" />
        </View>
        <Text className="nav-title">滤镜调节</Text>
      </View>

      {!hasResult ? (
        <View className="empty-state">
          <Text className="empty-text">暂无对比数据</Text>
          <Text className="empty-hint">请先完成去雾处理</Text>
        </View>
      ) : (
        <>
          {/* 实时预览区 */}
          <View className="preview-area">
            {isH5 ? (
              // H5 端：使用 CSS filter 直接作用于 Image
              <Image
                src={currentPreviewUrl}
                className="preview-image"
                mode="widthFix"
                style={showOrigin ? {} : { filter: filterStyle }}
                lazyLoad
              />
            ) : (
              // 小程序端：使用 Canvas 2D 渲染带滤镜的图片
              <Canvas
                type="2d"
                id="filterCanvas"
                className="preview-canvas"
                style={{
                  width: `${canvasDisplaySize.width}px`,
                  height: `${canvasDisplaySize.height}px`,
                }}
              />
            )}
            <View
              className="preview-toggle"
              onClick={() => setShowOrigin((prev) => !prev)}
            >
              <Text>{showOrigin ? "原图" : "滤镜效果"}</Text>
            </View>
          </View>

          {/* 滤镜参数控制区 */}
          <ScrollView className="filter-controls" scrollY>
            {/* 预设方案 */}
            <View className="preset-section">
              <View className="section-header">
                <Text className="section-title">预设方案</Text>
                <Text className="save-preset-btn" onClick={handleSavePreset}>
                  保存当前
                </Text>
              </View>
              <View className="preset-list">
                {BUILTIN_PRESETS.map((preset) => (
                  <View
                    key={preset.name}
                    className="preset-item"
                    onClick={() => applyPreset(preset.params)}
                  >
                    <Text>{preset.name}</Text>
                  </View>
                ))}
                {customPresets.map((preset, index) => (
                  <View
                    key={`custom-${index}`}
                    className="preset-item custom"
                    onClick={() => applyPreset(preset.params)}
                    onLongPress={() => handleDeletePreset(index)}
                  >
                    <Text>{preset.name}</Text>
                  </View>
                ))}
              </View>
            </View>

            {/* 滤镜参数滑动条 */}
            <View className="filter-sliders">
              <View className="section-header">
                <Text className="section-title">参数调节</Text>
                {hasChanges && (
                  <Text className="reset-btn" onClick={handleReset}>
                    重置
                  </Text>
                )}
              </View>
              {FILTER_CONFIGS.map((config) => (
                <View key={config.key} className="slider-item">
                  <View className="slider-label-row">
                    <Text className="slider-label">{config.label}</Text>
                    <Text className="slider-value">{params[config.key]}</Text>
                  </View>
                  <Slider
                    min={config.min}
                    max={config.max}
                    value={params[config.key]}
                    step={1}
                    activeColor="#1890ff"
                    onChanging={(e: any) =>
                      handleParamChange(config.key, e.detail.value)
                    }
                    onChange={(e: any) =>
                      handleParamChange(config.key, e.detail.value)
                    }
                  />
                </View>
              ))}
            </View>
          </ScrollView>
        </>
      )}

      {/* 底部工具栏 */}
      <CompareToolbar currentMode="filter" resultUrl={result?.resultUrl} />
    </View>
  );
};

export default FilterPage;
