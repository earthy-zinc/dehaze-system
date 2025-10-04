// 基本选择器
import { RootState } from "@/store";
import { createSelector } from "@reduxjs/toolkit";

const selectImageShowState = (state: RootState) => state.imageShow;

// 尺寸比例计算
export const selectScaleX = createSelector(
  [selectImageShowState],
  (state) => state.naturalWidth / state.width
);

export const selectScaleY = createSelector(
  [selectImageShowState],
  (state) => state.naturalHeight / state.height
);

// 遮罩层尺寸
export const selectMaskWidth = createSelector(
  [selectImageShowState],
  (state) => state.magnifier.width / state.magnifier.zoomLevel
);

export const selectMaskHeight = createSelector(
  [selectImageShowState],
  (state) => state.magnifier.height / state.magnifier.zoomLevel
);

// 生成按钮禁用状态
export const selectDisableGenerate = createSelector(
  [selectImageShowState],
  (state) => state.urls.length !== 1 || !state.modelId
);

// 其他选择器示例
export const selectMousePosition = createSelector(
  [selectImageShowState],
  (state) => ({
    x: state.mouse.x,
    y: state.mouse.y,
  })
);

export const selectMagnifierZoomLevel = createSelector(
  [selectImageShowState],
  (state) => state.magnifier.zoomLevel
);

export const selectDividerEnabled = createSelector(
  [selectImageShowState],
  (state) => state.divider.enabled
);
