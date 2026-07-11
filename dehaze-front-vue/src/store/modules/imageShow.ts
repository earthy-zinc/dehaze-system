import { ImageTypeEnum } from "@/enums/ImageTypeEnum";
import { useSettingsStore } from "@/store";

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
  const settingsStore = useSettingsStore();

  const loading = ref(false);

  const modelId = ref();

  const imageInfo = reactive({
    // 缩略图
    images: {
      urls: [] as ImageUrlType[],
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

  // 去雾算法参数（处理时使用，效果对比页用于参数对比）
  const dehazeParams = reactive({
    dehazeStrength: 50,
    colorSaturation: 50,
    contrast: 50,
    sharpen: 30,
  });

  const disableGenerate = computed(() => {
    return imageInfo.images.urls.length !== 1 || !modelId.value;
  });

  function setLoading(flag: boolean) {
    loading.value = flag;
  }

  function setModelId(id: number) {
    modelId.value = id;
  }

  function setImageUrls(urls: ImageUrlType[]) {
    imageInfo.images.urls = urls;
  }

  function setImageUrl(url: string, type: ImageTypeEnum) {
    const index = imageInfo.images.urls.findIndex(
      (item) => item.label.text === type
    );
    // 有雾图：背景黑，文字白
    // 预测图：背景主题色，文字白
    // 清晰图：背景蓝，文字黑
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
        backgroundColor: settingsStore.themeColor,
      };
    } else {
      id = 2;
      label = { text: type, color: "#000", backgroundColor: "#00f" };
    }
    if (index !== -1) {
      imageInfo.images.urls[index] = { id, label, url };
    } else {
      imageInfo.images.urls.push({ id, label, url });
    }
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
    loading,
    scaleX,
    scaleY,
    imageInfo,
    modelId,
    mouse,
    magnifierInfo,
    mask,
    maskWidth,
    maskHeight,
    dividerInfo,
    dehazeParams,
    disableGenerate,
    setLoading,
    setModelId,
    setImageUrls,
    setImageUrl,
    setImageNaturalSize,
    setImageSize,
    setBrightness,
    setContrast,
    setSaturate,
    setMagnifierShow,
    setDividerShow,
    toggleDividerShow,
    setMagnifierShape,
    setMagnifierSize,
    setMagnifierZoomLevel,
    setMaskXY,
    setMouseXY,
  };
});
