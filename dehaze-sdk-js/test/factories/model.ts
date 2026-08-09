import {
  BatchPredictionForm,
  CompareReportForm,
  EvaluationForm,
  PredictionForm,
  PresetForm,
} from "../../index";
import { DEHAZE_HOST } from "#/config/constant";

const DATASET_BASE_URL = `http://${DEHAZE_HOST}:9000/datasets`;

export function createPredictionForm(overrides: Partial<PredictionForm> = {}): PredictionForm {
  return {
    algorithmId: 1,
    imageUrl: `${DATASET_BASE_URL}/NH-HAZE-2023/hazy/001.JPG`,
    ...overrides,
  };
}

export function createEvaluationForm(overrides: Partial<EvaluationForm> = {}): EvaluationForm {
  return {
    algorithmId: 1,
    predUrl: `${DATASET_BASE_URL}/NH-HAZE-2023/hazy/001.JPG`,
    gtUrl: `${DATASET_BASE_URL}/NH-HAZE-2023/clean/001.JPG`,
    ...overrides,
  };
}

export function createBatchPredictionForm(
  overrides: Partial<BatchPredictionForm> = {}
): BatchPredictionForm {
  return {
    algorithmId: 1,
    items: [
      { imageUrl: `${DATASET_BASE_URL}/NH-HAZE-2023/hazy/001.JPG` },
      { imageUrl: `${DATASET_BASE_URL}/NH-HAZE-2023/hazy/002.JPG` },
    ],
    ...overrides,
  };
}

export function createPresetForm(overrides: Partial<PresetForm> = {}): PresetForm {
  return {
    name: `preset_${Date.now()}`,
    algorithmId: 1,
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
