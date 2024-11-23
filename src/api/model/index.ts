import request from "@/utils/requestPy";
import { EvalParam, EvalResult, PredParam } from "src/api/model/model";

class ModelAPI {
  static prediction(data: PredParam) {
    return request<any, string>({
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
