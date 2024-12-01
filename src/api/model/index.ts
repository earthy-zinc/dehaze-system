import request from "@/utils/requestPy";
import { EvalParam, EvalResult, PredParam, PredResult } from "./model";

class ModelAPI {
  static prediction(data: PredParam) {
    return request<any, PredResult>({
      url: "/model/prediction",
      method: "post",
      data,
    });
  }

  static evaluation(data: EvalParam) {
    return request<any, EvalResult[]>({
      url: "/model/evaluation",
      method: "post",
      data,
    });
  }
}

export default ModelAPI;
