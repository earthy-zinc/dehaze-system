import { PredictionForm, EvaluationForm } from "../../index";

const DATASET_BASE_URL = `http://${process.env.DEHAZE_HOST || "127.0.0.1"}:9000/datasets`;

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
