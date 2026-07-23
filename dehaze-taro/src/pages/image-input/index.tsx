/**
 * 图像输入页面
 * 支持四种输入方式：上传、拍照、样例库、历史记录
 */

import React, { useEffect } from "react";
import { View, Text } from "@tarojs/components";
import { Fire } from "@taroify/icons";

// 组件导入
import InputMethodTabs from "./components/InputMethodTabs";
import UploadArea from "./components/UploadArea";
import CameraArea from "./components/CameraArea";
import SampleGallery from "./components/SampleGallery";
import HistoryList from "./components/HistoryList";
import ImagePreview from "./components/ImagePreview";

// Store
import { ImageInputProvider, useImageInput } from "./store/imageInputStore";

import "./index.less";

// 主内容组件
const ImageInputContent: React.FC = () => {
  const {
    state,
    setActiveMethod,
    chooseImageFromAlbum,
    takePhoto,
    selectSampleImage,
    setSampleCategory,
    getSampleImages,
    loadHistory,
    deleteHistoryRecord,
    clearHistory,
    selectHistoryRecord,
    reprocessHistoryRecord,
    cancelSelection,
    confirmAndNavigate,
  } = useImageInput();

  // 切换到历史记录时加载数据
  useEffect(() => {
    if (state.activeMethod === "history") {
      loadHistory();
    }
  }, [state.activeMethod, loadHistory]);

  // 获取当前分类的样例图片（从 state 读取，由 store 异步加载）
  const sampleImages = getSampleImages(state.sampleCategory);

  // 快速体验 - 随机选择样例图片
  const handleQuickStart = () => {
    if (sampleImages.length > 0) {
      const randomIndex = Math.floor(Math.random() * sampleImages.length);
      selectSampleImage(sampleImages[randomIndex]);
    } else {
      // 样例数据尚未加载，切换到样例 tab 触发加载
      setActiveMethod("sample");
    }
  };

  return (
    <View className="image-input-page">
      {/* 页面标题 */}
      <View className="page-header">
        <Text className="page-title">图像输入</Text>
        <Text className="page-subtitle">选择图片开始去雾处理</Text>
      </View>

      {/* 输入方式选择 */}
      <InputMethodTabs
        activeMethod={state.activeMethod}
        onChange={setActiveMethod}
      />

      {/* 内容区域 */}
      <View className="content-area">
        {/* 上传区域 */}
        {state.activeMethod === "upload" && (
          <UploadArea
            onUpload={chooseImageFromAlbum}
            loading={state.uploadLoading}
            error={state.uploadError}
          />
        )}

        {/* 拍照区域 */}
        {state.activeMethod === "camera" && (
          <CameraArea onCapture={takePhoto} loading={state.uploadLoading} />
        )}

        {/* 样例图片库 */}
        {state.activeMethod === "sample" && (
          <SampleGallery
            samples={sampleImages}
            category={state.sampleCategory}
            loading={state.sampleLoading}
            onCategoryChange={setSampleCategory}
            onSelect={selectSampleImage}
          />
        )}

        {/* 历史记录 */}
        {state.activeMethod === "history" && (
          <HistoryList
            records={state.historyRecords}
            loading={state.historyLoading}
            onSelect={selectHistoryRecord}
            onReprocess={reprocessHistoryRecord}
            onDelete={deleteHistoryRecord}
            onClear={clearHistory}
          />
        )}
      </View>

      {/* 快速体验卡片 */}
      <View className="quick-start-card" onClick={handleQuickStart}>
        <View className="quick-start-content">
          <View className="quick-start-icon">
            <Fire size="24" color="white" />
          </View>
          <View className="quick-start-text">
            <Text className="quick-title">快速体验</Text>
            <Text className="quick-desc">使用样例图片快速体验去雾效果</Text>
          </View>
        </View>
        <View className="quick-btn">
          <Text>立即体验</Text>
        </View>
      </View>

      {/* 图片预览弹窗 */}
      <ImagePreview
        visible={state.previewVisible}
        imageData={state.currentImage}
        onConfirm={confirmAndNavigate}
        onCancel={cancelSelection}
      />
    </View>
  );
};

// 页面组件（包装 Provider）
const ImageInputPage: React.FC = () => {
  return (
    <ImageInputProvider>
      <ImageInputContent />
    </ImageInputProvider>
  );
};

export default ImageInputPage;
