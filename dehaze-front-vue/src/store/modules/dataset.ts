// 数据集小仓库
import { Dataset } from "@/api/dataset/model";
import DatasetAPI from "@/api/dataset";
import { DatasetQuery } from "@/api/dataset/model";

export const useDatasetStore = defineStore("dataset", () => {
  // 数据集列表
  const datasetList = ref<Dataset[]>([]);

  /**
   * 获取数据集列表
   * @param queryParams 查询参数
   */
  const getDatasetList = async (queryParams?: DatasetQuery) => {
    datasetList.value = await DatasetAPI.getList(queryParams);
  };

  /**
   * 新增数据集数据
   * @param data 数据集数据
   * @returns 新增结果（用于交互）
   */
  const addDataset = async (data: Dataset) => {
    return await DatasetAPI.add(data);
  };

  /**
   * 更新数据集数据
   * @param id 数据集id
   * @param data 数据集数据
   * @returns 更新结果（用于交互）
   */
  const updateDataset = async (id: number, data: Dataset) => {
    return await DatasetAPI.update(id, data);
  };

  /**
   * 删除数据集数据
   * @param ids 数据集id数组
   * @returns 删除结果（用于交互）
   */
  const deleteDatasetByIds = async (ids: string[]) => {
    return await DatasetAPI.deleteByIds(ids);
  };

  return {
    datasetList,
    getDatasetList,
    addDataset,
    updateDataset,
    deleteDatasetByIds,
  };
});
