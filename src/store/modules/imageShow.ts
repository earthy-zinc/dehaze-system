import { store } from "@/store";

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

export const useImageShowStore = defineStore("imageShow", () => {
  const imageInfo = reactive({
    // 缩略图
    images: {
      urls: [
        {
          id: 0,
          label: { text: "原图", color: "white", backgroundColor: "black" },
          url: "http://10.16.39.192:8989/api/v1/files/dataset/origin/NH-HAZE-2023/hazy/001.JPG",
        },
        {
          id: 1,
          label: {
            text: "对比图",
            color: "white",
            backgroundColor: "green",
          },
          url: "http://10.16.39.192:8989/api/v1/files/dataset/origin/NH-HAZE-2023/clean/001.JPG",
        },
        {
          id: 2,
          label: {
            text: "对比图",
            color: "white",
            backgroundColor: "green",
          },
          url: "http://10.16.39.192:8989/api/v1/files/dataset/origin/NH-HAZE-2023/hazy/001.JPG",
        },
      ] as ImageUrlType[],
      naturalWidth: 0,
      naturalHeight: 0,
    },
    // 图片实际宽高
    width: 0,
    height: 0,
    brightness: 100,
    contrast: 100,
    saturate: 100,
  });

  // 相对于图片原始分辨率缩放倍数
  const scaleX = computed(() => {
    return imageInfo.images.naturalWidth / imageInfo.width;
  });

  const scaleY = computed(() => {
    return imageInfo.images.naturalHeight / imageInfo.height;
  });

  const mouse = reactive({
    x: 0,
    y: 0,
  });

  const magnifierInfo = reactive({
    enabled: true,
    zoomLevel: 2,
    shape: "square",
    width: 100,
    height: 100,
  });

  const mask = reactive({
    x: 0,
    y: 0,
  });

  const maskWidth = computed(() => {
    return magnifierInfo.width / magnifierInfo.zoomLevel;
  });

  const maskHeight = computed(() => {
    return magnifierInfo.height / magnifierInfo.zoomLevel;
  });

  const dividerInfo = reactive({
    enabled: true,
  });

  function setImageUrls(urls: ImageUrlType[]) {
    imageInfo.images.urls = urls;
  }

  function setImageNaturalSize(width: number, height: number) {
    imageInfo.images.naturalWidth = width;
    imageInfo.images.naturalHeight = height;
  }

  function setImageSize(width: number, height: number) {
    imageInfo.width = width;
    imageInfo.height = height;
  }

  function setBrightness(brightness: number) {
    imageInfo.brightness = brightness;
  }

  function setContrast(contrast: number) {
    imageInfo.contrast = contrast;
  }

  function setSaturate(saturate: number) {
    imageInfo.saturate = saturate;
  }

  function setMagnifierShow(enabled: boolean) {
    magnifierInfo.enabled = enabled;
  }

  function setDividerShow(enabled: boolean) {
    dividerInfo.enabled = enabled;
  }

  function toggleMagnifierShow() {
    magnifierInfo.enabled = !magnifierInfo.enabled;
  }

  function toggleDividerShow() {
    dividerInfo.enabled = !dividerInfo.enabled;
  }

  function setMagnifierShape(shape: "circle" | "square") {
    magnifierInfo.shape = shape;
  }

  function setMagnifierSize(width: number, height: number) {
    magnifierInfo.width = width;
    magnifierInfo.height = height;
  }

  function setMagnifierZoomLevel(zoomLevel: number) {
    magnifierInfo.zoomLevel = zoomLevel;
  }

  function setMaskXY(x: number, y: number) {
    mask.x = x;
    mask.y = y;
  }

  function setMouseXY(x: number, y: number) {
    mouse.x = x;
    mouse.y = y;
  }

  return {
    scaleX,
    scaleY,
    imageInfo,
    mouse,
    magnifierInfo,
    mask,
    maskWidth,
    maskHeight,
    dividerInfo,
    setImageUrls,
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
  };
});

export function useImageShowHook() {
  return useImageShowStore(store);
}
