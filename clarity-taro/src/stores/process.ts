import { create } from "zustand";
import type { ImageTaskResponse } from "clarity-sdk-js";

export interface ProcessImage {
  url: string;
  name: string;
  width?: number;
  height?: number;
  size?: number;
  cleanUrl?: string;
}

/** 当前选中的图像模型（统一模型规范） */
export interface SelectedModel {
  /** 模型标识（提交图像任务时使用） */
  modelId: string;
  displayName: string;
  /** 推荐来源记录 ID（从推荐列表选中时携带，用于追踪推荐采纳率） */
  recommendedBy?: number;
}

interface ProcessState {
  image: ProcessImage | null;
  model: SelectedModel | null;
  result: ImageTaskResponse | null;
  setImage: (image: ProcessImage) => void;
  setModel: (model: SelectedModel) => void;
  setResult: (result: ImageTaskResponse) => void;
  reset: () => void;
}

export const useProcessStore = create<ProcessState>()((set) => ({
  image: null,
  model: null,
  result: null,
  setImage: (image) => set({ image }),
  setModel: (model) => set({ model }),
  setResult: (result) => set({ result }),
  reset: () => set({ image: null, model: null, result: null }),
}));
