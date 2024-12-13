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

const initialState = {
  loading: false,
  modelId: 0,
  imageInfo: {
    images: {
      urls: [
        {
          id: 1,
          label: {
            text: ImageTypeEnum.CLEAN,
            color: "#000",
            backgroundColor: "#fff",
          },
          url: "http://10.16.39.192:8989/api/v1/files/dataset/origin/NH-HAZE-2021/clean/041.png",
        },
        {
          id: 2,
          label: {
            text: ImageTypeEnum.PRED,
            color: "#fff",
            backgroundColor: "#4a4",
          },
          url: "http://10.16.39.192:8989/api/v1/files/dataset/origin/NH-HAZE-2021/hazy/041.png",
        },
      ] as ImageUrlType[],
      naturalWidth: 0,
      naturalHeight: 0,
    },
    width: 0,
    height: 0,
    brightness: 100,
    contrast: 100,
    saturate: 100,
  },
  mouse: {
    x: 0,
    y: 0,
  },
  magnifierInfo: {
    enabled: true,
    zoomLevel: 2,
    shape: "square",
    width: 100,
    height: 100,
  },
  mask: {
    x: 0,
    y: 0,
  },
  dividerInfo: {
    enabled: true,
  },
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
    setImageUrls: (state, action: PayloadAction<ImageUrlType[]>) => {
      state.imageInfo.images.urls = action.payload;
    },
    setImageUrl: (
      state,
      action: PayloadAction<{ url: string; type: ImageTypeEnum }>
    ) => {
      const { url, type } = action.payload;
      const index = state.imageInfo.images.urls.findIndex(
        (item) => item.label.text === type
      );
      let label;
      let id;
      if (type === ImageTypeEnum.HAZE) {
        id = 0;
        label = { text: type, color: "#fff", backgroundColor: "#000" };
      } else if (type === ImageTypeEnum.PRED) {
        id = 1;
        label = {
          text: type,
          color: "#fff",
          backgroundColor: "#4a4", // Assuming themeColor is "#4a4"
        };
      } else {
        id = 2;
        label = { text: type, color: "#000", backgroundColor: "#00f" };
      }
      if (index !== -1) {
        state.imageInfo.images.urls[index] = { id, label, url };
      } else {
        state.imageInfo.images.urls.push({ id, label, url });
      }
    },
    updateImageUrls: (state, action: PayloadAction<ImageUrlType>) => {
      const { payload } = action;
      const index = state.imageInfo.images.urls.findIndex(
        (item) => item.label.text === payload.label.text
      );
      if (index !== -1) {
        state.imageInfo.images.urls[index] = payload;
      } else {
        state.imageInfo.images.urls.push(payload);
      }
    },
    setImageNaturalSize: (
      state,
      action: PayloadAction<{ width: number; height: number }>
    ) => {
      const { width, height } = action.payload;
      state.imageInfo.images.naturalWidth = width;
      state.imageInfo.images.naturalHeight = height;
    },
    setImageSize: (
      state,
      action: PayloadAction<{ width: number; height: number }>
    ) => {
      const { width, height } = action.payload;
      state.imageInfo.width = width;
      state.imageInfo.height = height;
    },
    setBrightness: (state, action: PayloadAction<number>) => {
      state.imageInfo.brightness = action.payload;
    },
    setContrast: (state, action: PayloadAction<number>) => {
      state.imageInfo.contrast = action.payload;
    },
    setSaturate: (state, action: PayloadAction<number>) => {
      state.imageInfo.saturate = action.payload;
    },
    setMagnifierShow: (state, action: PayloadAction<boolean>) => {
      state.magnifierInfo.enabled = action.payload;
    },
    setDividerShow: (state, action: PayloadAction<boolean>) => {
      state.dividerInfo.enabled = action.payload;
    },
    toggleMagnifierShow: (state) => {
      state.magnifierInfo.enabled = !state.magnifierInfo.enabled;
    },
    toggleDividerShow: (state) => {
      state.dividerInfo.enabled = !state.dividerInfo.enabled;
    },
    setMagnifierShape: (state, action: PayloadAction<"circle" | "square">) => {
      state.magnifierInfo.shape = action.payload;
    },
    setMagnifierSize: (
      state,
      action: PayloadAction<{ width: number; height: number }>
    ) => {
      const { width, height } = action.payload;
      state.magnifierInfo.width = width;
      state.magnifierInfo.height = height;
    },
    setMagnifierZoomLevel: (state, action: PayloadAction<number>) => {
      state.magnifierInfo.zoomLevel = action.payload;
    },
    setMaskXY: (state, action: PayloadAction<{ x: number; y: number }>) => {
      const { x, y } = action.payload;
      state.mask.x = x;
      state.mask.y = y;
    },
    setMouseXY: (state, action: PayloadAction<{ x: number; y: number }>) => {
      const { x, y } = action.payload;
      state.mouse.x = x;
      state.mouse.y = y;
    },
  },
});

const imageShowPersistConfig = {
  key: "imageShow",
  storage,
  whitelist: [
    "loading",
    "modelId",
    "imageInfo",
    "mouse",
    "magnifierInfo",
    "mask",
    "dividerInfo",
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
  setDividerShow,
  toggleMagnifierShow,
  toggleDividerShow,
  setMagnifierShape,
  setMagnifierSize,
  setMagnifierZoomLevel,
  setMaskXY,
  setMouseXY,
} = imageShowSlice.actions;

export default persistReducer(imageShowPersistConfig, imageShowSlice.reducer);
