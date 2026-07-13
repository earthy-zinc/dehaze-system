import { PredictionForm, EvaluationForm } from "../../index";

export function createPredictionForm(overrides: Partial<PredictionForm> = {}): PredictionForm {
  return {
    algorithmId: 1,
    imageUrl: "/api/v1/files/download/test_haze_image.jpg",
    ...overrides,
  };
}

export function createEvaluationForm(overrides: Partial<EvaluationForm> = {}): EvaluationForm {
  return {
    algorithmId: 1,
    predUrl: "/api/v1/files/download/test_pred.jpg",
    gtUrl: "/api/v1/files/download/test_gt.jpg",
    ...overrides,
  };
}
