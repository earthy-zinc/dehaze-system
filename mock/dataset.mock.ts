import { defineMock } from "./base";
import datasetVO from "./data/DatasetVO";
export default defineMock([
  {
    url: "dataset",
    method: ["GET"],
    body: (request) => {
      return {
        code: "00000",
        data: datasetVO,
        msg: "一切ok",
      };
    },
  },
  {
    url: "dataset/:id/form",
    method: ["GET"],
    body: {
      code: "00000",
      data: [],
      msg: "一切ok",
    },
  },
]);
