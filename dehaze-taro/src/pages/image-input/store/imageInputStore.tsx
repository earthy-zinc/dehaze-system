/**
 * 图像输入模块状态管理（zustand）
 */

import { create } from "zustand";
import { useCallback, useRef, useEffect } from "react";
import Taro from "@tarojs/taro";
import { confirmDialog } from "@/utils/dialog";
import { getErrorMessage } from "@/utils/error";
import { useProcessStore } from "@/stores/process";
import type { InputHistoryVO } from "dehaze-sdk-js";
import {
  InputMethod,
  SampleCategory,
  ImageData,
  SampleImage,
  ErrorCodes,
} from "../services/types";
import { ImageInputService, isImageInputError } from "../services/imageInput";
import {
  getHistoryPage,
  deleteHistoryRecord as deleteHistoryRecordService,
  clearAllHistory,
} from "../services/history";
import { fetchSampleImages } from "../services/sampleData";

/** 从 URL 中提取文件名（去除查询参数），失败时返回 fallback */
function extractFilenameFromUrl(url: string, fallback = "历史图片"): string {
  if (!url) return fallback;
  const path = url.split("?")[0];
  const segments = path.split("/");
  return segments[segments.length - 1] || fallback;
}

// 状态类型定义
interface ImageInputState {
  // 当前输入方式
  activeMethod: InputMethod;

  // 当前选中的图片
  currentImage: ImageData | null;

  // 上传状态
  uploadLoading: boolean;
  uploadError: string | null;

  // 样例图片
  sampleCategory: SampleCategory;
  sampleLoading: boolean;
  sampleImages: SampleImage[];

  // 历史记录
  historyRecords: InputHistoryVO[];
  historyLoading: boolean;

  // 预览弹窗
  previewVisible: boolean;
}

interface ImageInputStore extends ImageInputState {
  setActiveMethod: (method: InputMethod) => void;
  setCurrentImage: (image: ImageData | null) => void;
  handleError: (error: unknown, fallbackMsg: string) => void;
  chooseImageFromAlbum: () => Promise<void>;
  takePhoto: () => Promise<void>;
  selectSampleImage: (sample: SampleImage) => Promise<void>;
  setSampleCategory: (category: SampleCategory) => void;
  loadSampleImages: (category: SampleCategory) => Promise<void>;
  loadHistory: () => Promise<void>;
  deleteHistoryRecord: (id: number) => Promise<void>;
  clearHistory: () => Promise<void>;
  cancelSelection: () => void;
  confirmAndNavigate: () => void;
  reprocessHistoryRecord: (record: InputHistoryVO) => void;
}

// 初始状态
const initialState: ImageInputState = {
  activeMethod: "upload",
  currentImage: null,
  uploadLoading: false,
  uploadError: null,
  sampleCategory: "all",
  sampleLoading: false,
  sampleImages: [],
  historyRecords: [],
  historyLoading: false,
  previewVisible: false,
};

export const useImageInputStore = create<ImageInputStore>()((set, get) => ({
  ...initialState,

  setActiveMethod: (method) => set({ activeMethod: method }),

  setCurrentImage: (image) => {
    set({ currentImage: image });
    if (image) {
      set({ previewVisible: true });
    }
  },

  // 统一错误处理
  handleError: (error: unknown, fallbackMsg: string) => {
    if (isImageInputError(error) && error.code === ErrorCodes.USER_CANCEL) {
      // 用户取消，不提示
      return;
    }
    const msg = isImageInputError(error)
      ? error.message
      : getErrorMessage(error, fallbackMsg);
    set({ uploadError: msg });
    Taro.showToast({ title: msg, icon: "none" });
  },

  // 从相册选择图片
  chooseImageFromAlbum: async () => {
    try {
      set({ uploadLoading: true, uploadError: null });

      const tempFiles = await ImageInputService.chooseImage(1);
      if (tempFiles.length > 0) {
        const imageData = await ImageInputService.processImageFile(
          tempFiles[0]
        );
        get().setCurrentImage(imageData);
      }
    } catch (error: unknown) {
      get().handleError(error, "选择图片失败");
    } finally {
      set({ uploadLoading: false });
    }
  },

  // 拍照
  takePhoto: async () => {
    try {
      set({ uploadLoading: true, uploadError: null });

      // 检查相机权限
      const hasPermission = await ImageInputService.checkCameraPermission();
      if (!hasPermission) {
        const granted = await ImageInputService.requestCameraPermission();
        if (!granted) {
          throw new Error("相机权限被拒绝");
        }
      }

      const tempFile = await ImageInputService.takePhoto();
      const imageData = await ImageInputService.processImageFile(tempFile);
      get().setCurrentImage(imageData);
    } catch (error: unknown) {
      get().handleError(error, "拍照失败");
    } finally {
      set({ uploadLoading: false });
    }
  },

  // 选择样例图片
  selectSampleImage: async (sample) => {
    try {
      set({ sampleLoading: true });

      const imageData = await ImageInputService.loadImageFromUrl(
        sample.url,
        sample.name
      );
      get().setCurrentImage({
        ...imageData,
        sampleInfo: sample,
        cleanUrl: sample.cleanUrl,
      });

      Taro.showToast({ title: "样例图片加载成功", icon: "success" });
    } catch (error: unknown) {
      get().handleError(error, "加载失败");
    } finally {
      set({ sampleLoading: false });
    }
  },

  // 切换样例分类（触发重新加载）
  setSampleCategory: (category) => set({ sampleCategory: category }),

  // 加载样例图片
  loadSampleImages: async (category) => {
    try {
      set({ sampleLoading: true });
      const images = await fetchSampleImages(category);
      set({ sampleImages: images });
    } catch (error: unknown) {
      console.error("加载样例图片失败:", error);
      set({ sampleImages: [] });
    } finally {
      set({ sampleLoading: false });
    }
  },

  // 加载历史记录
  loadHistory: async () => {
    try {
      set({ historyLoading: true });
      const { list } = await getHistoryPage();
      set({ historyRecords: list, historyLoading: false });
    } catch (error: unknown) {
      console.error("加载历史记录失败:", error);
      set({ historyRecords: [], historyLoading: false });
    }
  },

  // 删除历史记录
  deleteHistoryRecord: async (id) => {
    try {
      await deleteHistoryRecordService(id);
      set({
        historyRecords: get().historyRecords.filter((r) => r.id !== id),
      });
      Taro.showToast({ title: "已删除", icon: "success" });
    } catch (error: unknown) {
      Taro.showToast({ title: "删除失败", icon: "none" });
    }
  },

  // 清空历史记录
  clearHistory: async () => {
    const confirmed = await confirmDialog({
      title: "确认清空",
      content: "确定要清空所有历史记录吗？",
      confirmColor: "#ef4444",
    });
    if (!confirmed) return;
    try {
      await clearAllHistory();
      set({ historyRecords: [] });
      Taro.showToast({ title: "已清空", icon: "success" });
    } catch (error: unknown) {
      Taro.showToast({ title: "清空失败", icon: "none" });
    }
  },

  // 取消选择
  cancelSelection: () => {
    set({ currentImage: null, previewVisible: false });
  },

  // 确认选择，跳转算法选择页
  confirmAndNavigate: () => {
    const currentImage = get().currentImage;
    if (!currentImage) {
      Taro.showToast({ title: "请先选择图片", icon: "none" });
      return;
    }

    // 保存当前图片到全局状态
    useProcessStore.getState().setImage(currentImage);

    // 跳转到算法选择页面
    Taro.navigateTo({
      url: "/pages/algorithm-select/index",
    });
  },

  // 重新处理历史记录（设计文档：历史记录 → 更换算法 → 重新处理）
  reprocessHistoryRecord: (record) => {
    const url = record.originalImageUrl || "";
    if (!url) {
      Taro.showToast({ title: "原图地址缺失", icon: "none" });
      return;
    }
    const filename = extractFilenameFromUrl(url);
    useProcessStore.getState().setImage({
      url,
      width: 0,
      height: 0,
      size: 0,
      name: filename,
    });
    Taro.navigateTo({ url: "/pages/algorithm-select/index" });
  },
}));

/**
 * 兼容 hook：订阅 zustand store，返回 { state, ...actions }，
 * 与原 Context + Reducer 的 useImageInput 返回结构保持一致。
 */
export function useImageInput() {
  const state = useImageInputStore();
  const actions = useImageInputStore.getState();

  // 用 ref 跟踪已加载过的样例分类，避免重复请求
  const loadedSampleCategoryRef = useRef<SampleCategory | null>(null);

  // 当分类变化时重新加载样例图片
  useEffect(() => {
    if (
      state.activeMethod === "sample" &&
      loadedSampleCategoryRef.current !== state.sampleCategory
    ) {
      actions.loadSampleImages(state.sampleCategory);
    }
  }, [state.sampleCategory, state.activeMethod, actions.loadSampleImages]);

  // 获取当前分类的样例图片（从 state 读取）
  const getSampleImages = useCallback(
    (category: SampleCategory): SampleImage[] => {
      if (category === "all") return state.sampleImages;
      return state.sampleImages.filter((s) => s.category === category);
    },
    [state.sampleImages]
  );

  // 选择历史记录
  const selectHistoryRecord = useCallback(
    (record: InputHistoryVO) => {
      // 从历史记录加载原图（后端 VO 未返回尺寸/大小，仅传递 URL 与名称）
      const url = record.originalImageUrl || "";
      const filename = extractFilenameFromUrl(url);
      actions.setCurrentImage({
        url,
        path: url,
        width: 0,
        height: 0,
        size: 0,
        name: filename,
      });
    },
    [actions.setCurrentImage]
  );

  return {
    state,
    // Actions
    setActiveMethod: actions.setActiveMethod,
    setCurrentImage: actions.setCurrentImage,
    chooseImageFromAlbum: actions.chooseImageFromAlbum,
    takePhoto: actions.takePhoto,
    selectSampleImage: actions.selectSampleImage,
    setSampleCategory: actions.setSampleCategory,
    getSampleImages,
    loadHistory: actions.loadHistory,
    deleteHistoryRecord: actions.deleteHistoryRecord,
    clearHistory: actions.clearHistory,
    selectHistoryRecord,
    reprocessHistoryRecord: actions.reprocessHistoryRecord,
    cancelSelection: actions.cancelSelection,
    confirmAndNavigate: actions.confirmAndNavigate,
  };
}
