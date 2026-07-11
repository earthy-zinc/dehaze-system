// 数据集小仓库
import {
  Dataset,
  DatasetAPI,
  DatasetAddForm,
  DatasetQuery,
  DatasetUpdateForm,
} from "dehaze-sdk-js";

export const useDatasetStore = defineStore("dataset", () => {
  // 数据集列表
  const datasetList = ref<Dataset[]>([]);

  /**
   * 获取数据集列表
   * @param queryParams 查询参数
   */
  const getDatasetList = async (queryParams?: DatasetQuery) => {
    datasetList.value = (await DatasetAPI.getList(queryParams)).list;
  };

  /**
   * 新增数据集数据
   * @param data 数据集数据
   * @returns 新增结果（用于交互）
   */
  const addDataset = async (data: DatasetAddForm) => {
    return await DatasetAPI.add(data);
  };

  /**
   * 更新数据集数据
   * @param id 数据集id
   * @param data 数据集数据
   * @returns 更新结果（用于交互）
   */
  const updateDataset = async (id: number, data: DatasetUpdateForm) => {
    return await DatasetAPI.update(id, data);
  };

  /**
   * 删除数据集数据
   * @param id 数据集id
   * @returns 删除结果（用于交互）
   */
  const deleteDataset = async (id: number) => {
    return await DatasetAPI.deleteById(id);
  };

  return {
    datasetList,
    getDatasetList,
    addDataset,
    updateDataset,
    deleteDataset,
  };
});
