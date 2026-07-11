// 模型小仓库
import { Algorithm, AlgorithmAPI, AlgorithmQuery } from "dehaze-sdk-js";

export const useAlgorithmStore = defineStore("algorithm", () => {
  // 模型列表
  const algorithmList = ref<Algorithm[]>([]);
  // 模型下拉框选项列表
  const algorithmOptions = ref<OptionType[]>([]);

  /**
   * 获取模型列表
   * @param queryParams 查询参数
   */
  const getAlgorithmList = async (queryParams?: AlgorithmQuery) => {
    algorithmList.value = await AlgorithmAPI.getList(queryParams);
  };

  /**
   * 获取模型下拉框选项列表
   */
  const getAlgorithmOptions = async () => {
    algorithmOptions.value = await AlgorithmAPI.getOption();
  };

  /**
   * 新增模型数据
   * @param data 模型数据
   * @returns 新增结果（用于交互）
   */
  const addAlgorithm = async (data: Algorithm) => {
    return await AlgorithmAPI.add(data);
  };

  /**
   * 更新模型数据
   * @param id 模型id
   * @param data 模型数据
   * @returns 更新结果（用于交互）
   */
  const updateAlgorithm = async (id: number, data: Algorithm) => {
    return await AlgorithmAPI.update(id, data);
  };

  /**
   * 更新算法状态
   * @param id 算法id
   * @param status 状态(1:启用；0:禁用)
   * @returns 更新结果（用于交互）
   */
  const updateAlgorithmStatus = async (id: number, status: number) => {
    return await AlgorithmAPI.updateStatus(id, status);
  };

  /**
   * 删除模型数据
   * @param ids 模型id数组
   * @returns 删除结果（用于交互）
   */
  const deleteAlgorithmByIds = async (ids: string[]) => {
    return await AlgorithmAPI.deleteByIds(ids);
  };

  return {
    algorithmList,
    algorithmOptions,
    getAlgorithmList,
    getAlgorithmOptions,
    addAlgorithm,
    updateAlgorithm,
    updateAlgorithmStatus,
    deleteAlgorithmByIds,
  };
});
