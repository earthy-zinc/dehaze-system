import { CompareReportForm, EvaluationForm, ImageEditForm, PresetForm } from "../../index";
import { NGINX_STATIC_HOST, NGINX_STATIC_PORT } from "#/config/constant";

const DATASET_BASE_URL = `http://${NGINX_STATIC_HOST}:${NGINX_STATIC_PORT}/datasets`;

/** 默认图像模型标识（阶段1 迁移映射：算法 id=1 AECR-NET → aecrnet） */
export const DEFAULT_IMAGE_MODEL = "aecrnet";

export function createImageEditForm(overrides: Partial<ImageEditForm> = {}): ImageEditForm {
  return {
    model: DEFAULT_IMAGE_MODEL,
    imageUrl: `${DATASET_BASE_URL}/NH-HAZE-2023/hazy/001.JPG`,
    ...overrides,
  };
}

export function createEvaluationForm(overrides: Partial<EvaluationForm> = {}): EvaluationForm {
  return {
    modelId: DEFAULT_IMAGE_MODEL,
    predUrl: `${DATASET_BASE_URL}/NH-HAZE-2023/hazy/001.JPG`,
    gtUrl: `${DATASET_BASE_URL}/NH-HAZE-2023/clean/001.JPG`,
    ...overrides,
  };
}

export function createPresetForm(overrides: Partial<PresetForm> = {}): PresetForm {
  return {
    name: `preset_${Date.now()}`,
    modelId: DEFAULT_IMAGE_MODEL,
    params: JSON.stringify({ gamma: 1.0, clipLimit: 2.0 }),
    ...overrides,
  };
}

export function createCompareReportForm(
  overrides: Partial<CompareReportForm> = {}
): CompareReportForm {
  return {
    logId: 1,
    ...overrides,
  };
}
