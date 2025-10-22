import { ImageTypeEnum } from "@/enums/ImageTypeEnum";
import { createSlice, PayloadAction } from "@reduxjs/toolkit";
import { persistReducer } from "redux-persist";
import storage from "redux-persist/lib/storage";

export interface ImageUrlType {
  id: number;
  label: LabelType;
  url: string;
}

export interface LabelType {
  text: string;
  color: string;
  backgroundColor: string;
}

export interface ImageShowState {
  loading: boolean;
  modelId: number;
  urls: ImageUrlType[];
  naturalWidth: number;
  naturalHeight: number;
  width: number;
  height: number;
  brightness: number;
  contrast: number;
  saturate: number;
  magnifier: {
    enabled: boolean;
    zoomLevel: number;
    shape: "circle" | "square";
    width: number;
    height: number;
  };
  mask: { x: number; y: number };
  divider: { enabled: boolean };
  mouse: { x: number; y: number };
}

const initialState: ImageShowState = {
  loading: false,
  modelId: 0,
  urls: [
    {
      id: 1,
      label: {
        text: ImageTypeEnum.CLEAN,
        color: "#000",
        backgroundColor: "#fff",
      },
      url: "http://...",
    },
  ],
  naturalWidth: 0,
  naturalHeight: 0,
  width: 0,
  height: 0,
  brightness: 100,
  contrast: 100,
  saturate: 100,
  magnifier: {
    enabled: true,
    zoomLevel: 2,
    shape: "square",
    width: 100,
    height: 100,
  },
  mask: { x: 0, y: 0 },
  divider: { enabled: true },
  mouse: { x: 0, y: 0 },
};

export const imageShowSlice = createSlice({
  name: "imageShow",
  initialState,
  reducers: {
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.loading = action.payload;
    },
    setModelId: (state, action: PayloadAction<number>) => {
      state.modelId = action.payload;
    },
    // 图片 URL 相关
    setImageUrls: (state, action: PayloadAction<ImageUrlType[]>) => {
      state.urls = action.payload;
    },
    setImageUrl: (
      state,
      action: PayloadAction<{ url: string; type: ImageTypeEnum }>
    ) => {
      const { url, type } = action.payload;
      let id: number;
      let label: LabelType;
      if (type === ImageTypeEnum.HAZE) {
        id = 0;
        label = { text: type, color: "#fff", backgroundColor: "#000" };
      } else if (type === ImageTypeEnum.PRED) {
        id = 1;
        label = {
          text: type,
          color: "#fff",
          backgroundColor: "#4a4",
        };
      } else {
        id = 2;
        label = { text: type, color: "#000", backgroundColor: "#00f" };
      }
      const existing = state.urls.find((item) => item.label.text === type);
      if (existing) {
        existing.url = url;
        existing.label = label;
        existing.id = id;
      } else {
        state.urls.push({ id, label, url });
      }
    },
    updateImageUrls: (state, action: PayloadAction<ImageUrlType>) => {
      const { payload } = action;
      const index = state.urls.findIndex(
        (item) => item.label.text === payload.label.text
      );
      if (index !== -1) {
        state.urls[index] = payload;
      } else {
        state.urls.push(payload);
      }
    },

    // 图片尺寸相关
    setImageNaturalSize: (
      state,
      action: PayloadAction<{ width: number; height: number }>
    ) => {
      state.naturalWidth = action.payload.width;
      state.naturalHeight = action.payload.height;
    },
    setImageSize: (
      state,
      action: PayloadAction<{ width: number; height: number }>
    ) => {
      state.width = action.payload.width;
      state.height = action.payload.height;
    },

    // 调整参数
    setBrightness: (state, action: PayloadAction<number>) => {
      state.brightness = action.payload;
    },
    setContrast: (state, action: PayloadAction<number>) => {
      state.contrast = action.payload;
    },
    setSaturate: (state, action: PayloadAction<number>) => {
      state.saturate = action.payload;
    },

    // 放大镜相关
    setMagnifierShow: (state, action: PayloadAction<boolean>) => {
      state.magnifier.enabled = action.payload;
    },
    setMagnifierShape: (state, action: PayloadAction<"circle" | "square">) => {
      state.magnifier.shape = action.payload;
    },
    setMagnifierSize: (
      state,
      action: PayloadAction<{ width: number; height: number }>
    ) => {
      state.magnifier.width = action.payload.width;
      state.magnifier.height = action.payload.height;
    },
    setMagnifierZoomLevel: (state, action: PayloadAction<number>) => {
      state.magnifier.zoomLevel = action.payload;
    },

    // 遮罩层
    setMaskXY: (state, action: PayloadAction<{ x: number; y: number }>) => {
      state.mask.x = action.payload.x;
      state.mask.y = action.payload.y;
    },

    // 鼠标坐标
    setMouseXY: (state, action: PayloadAction<{ x: number; y: number }>) => {
      state.mouse.x = action.payload.x;
      state.mouse.y = action.payload.y;
    },

    // 分割线
    setDividerShow: (state, action: PayloadAction<boolean>) => {
      state.divider.enabled = action.payload;
    },
    toggleMagnifierShow: (state) => {
      state.magnifier.enabled = !state.magnifier.enabled;
    },
    toggleDividerShow: (state) => {
      state.divider.enabled = !state.divider.enabled;
    },
  },
});

const imageShowPersistConfig = {
  key: "imageShow",
  storage,
  whitelist: [
    "loading",
    "modelId",
    "urls",
    "naturalWidth",
    "naturalHeight",
    "width",
    "height",
    "magnifier",
    "mask",
    "divider",
    "mouse",
  ],
};

export const {
  setLoading,
  setModelId,
  setImageUrls,
  setImageUrl,
  updateImageUrls,
  setImageNaturalSize,
  setImageSize,
  setBrightness,
  setContrast,
  setSaturate,
  setMagnifierShow,
  setMagnifierShape,
  setMagnifierSize,
  setMagnifierZoomLevel,
  setMaskXY,
  setMouseXY,
  toggleMagnifierShow,
  toggleDividerShow,
  setDividerShow,
} = imageShowSlice.actions;

export default persistReducer(imageShowPersistConfig, imageShowSlice.reducer);
