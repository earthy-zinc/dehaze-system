// 算法模型 store
import { AlgorithmAPI, OptionType } from "dehaze-sdk-js";

export const useAlgorithmStore = defineStore("algorithm", () => {
  // 模型下拉框选项列表
  const algorithmOptions = ref<OptionType[]>([]);

  /** 获取模型下拉框选项列表 */
  const getAlgorithmOptions = async () => {
    algorithmOptions.value = await AlgorithmAPI.getOption();
  };

  return {
    algorithmOptions,
    getAlgorithmOptions,
  };
});
