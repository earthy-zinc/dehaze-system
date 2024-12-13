// 基本选择器
import { RootState } from "@/store";
import { createSelector } from "@reduxjs/toolkit";

const selectImageShowState = (state: RootState) => state.imageShow;

// 计算选择器
export const selectScaleX = createSelector(
  [selectImageShowState],
  (state) => state.imageInfo.images.naturalWidth / state.imageInfo.width
);

export const selectScaleY = createSelector(
  [selectImageShowState],
  (state) => state.imageInfo.images.naturalHeight / state.imageInfo.height
);

export const selectMaskWidth = createSelector(
  [selectImageShowState],
  (state) => state.magnifierInfo.width / state.magnifierInfo.zoomLevel
);

export const selectMaskHeight = createSelector(
  [selectImageShowState],
  (state) => state.magnifierInfo.height / state.magnifierInfo.zoomLevel
);

export const selectDisableGenerate = createSelector(
  [selectImageShowState],
  (state) => state.imageInfo.images.urls.length !== 1 || !state.modelId
);
