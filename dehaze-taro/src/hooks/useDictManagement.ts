import { useState, useCallback } from "react";
import Taro from "@tarojs/taro";
import type {
  DictTypeQuery,
  DictTypePageVO,
  DictTypeForm,
  DictQuery,
  DictPageVO,
  DictForm,
  OptionType,
} from "dehaze-sdk-js";

export const useDictManagement = () => {
  // 字典类型状态
  const [dictTypes, setDictTypes] = useState<DictTypePageVO[]>([]);
  const [dictTypeLoading, setDictTypeLoading] = useState(false);
  const [dictTypeTotal, setDictTypeTotal] = useState(0);
  const [dictTypeQueryParams, setDictTypeQueryParams] = useState<DictTypeQuery>(
    {
      pageNum: 1,
      pageSize: 10,
    }
  );

  // 字典数据状态
  const [dictItems, setDictItems] = useState<DictPageVO[]>([]);
  const [dictItemLoading, setDictItemLoading] = useState(false);
  const [dictItemTotal, setDictItemTotal] = useState(0);
  const [dictItemQueryParams, setDictItemQueryParams] = useState<DictQuery>({
    pageNum: 1,
    pageSize: 10,
  });

  // 获取字典类型分页列表
  const fetchDictTypes = useCallback(
    async (params?: Partial<DictTypeQuery>) => {
      setDictTypeLoading(true);
      try {
        const { DictAPI } = await import("dehaze-sdk-js");
        const query = { ...dictTypeQueryParams, ...params };
        const response = await DictAPI.getDictTypePage(query);
        setDictTypes(response.list);
        setDictTypeTotal(response.total);
        setDictTypeQueryParams(query);
        return response;
      } catch (error) {
        console.error("获取字典类型列表失败:", error);
        Taro.showToast({ title: "获取字典类型列表失败", icon: "none" });
        throw error;
      } finally {
        setDictTypeLoading(false);
      }
    },
    [dictTypeQueryParams]
  );

  // 新增字典类型
  const createDictType = useCallback(
    async (data: DictTypeForm) => {
      try {
        const { DictAPI } = await import("dehaze-sdk-js");
        await DictAPI.addDictType(data);
        await fetchDictTypes();
        Taro.showToast({ title: "新增字典类型成功", icon: "none" });
        return true;
      } catch (error) {
        console.error("新增字典类型失败:", error);
        Taro.showToast({ title: "新增字典类型失败", icon: "none" });
        throw error;
      }
    },
    [fetchDictTypes]
  );

  // 获取字典类型表单数据
  const fetchDictTypeForm = useCallback(async (id: number) => {
    try {
      const { DictAPI } = await import("dehaze-sdk-js");
      return await DictAPI.getDictTypeForm(id);
    } catch (error) {
      console.error("获取字典类型表单数据失败:", error);
      Taro.showToast({ title: "获取字典类型表单数据失败", icon: "none" });
      throw error;
    }
  }, []);

  // 修改字典类型
  const updateDictType = useCallback(
    async (id: number, data: DictTypeForm) => {
      try {
        const { DictAPI } = await import("dehaze-sdk-js");
        await DictAPI.updateDictType(id, data);
        await fetchDictTypes();
        Taro.showToast({ title: "修改字典类型成功", icon: "none" });
        return true;
      } catch (error) {
        console.error("修改字典类型失败:", error);
        Taro.showToast({ title: "修改字典类型失败", icon: "none" });
        throw error;
      }
    },
    [fetchDictTypes]
  );

  // 删除字典类型
  const deleteDictTypes = useCallback(
    async (ids: string) => {
      try {
        const { DictAPI } = await import("dehaze-sdk-js");
        await DictAPI.deleteDictTypes(ids);
        await fetchDictTypes();
        Taro.showToast({ title: "删除字典类型成功", icon: "none" });
        return true;
      } catch (error) {
        console.error("删除字典类型失败:", error);
        Taro.showToast({ title: "删除字典类型失败", icon: "none" });
        throw error;
      }
    },
    [fetchDictTypes]
  );

  // 搜索字典类型
  const searchDictTypes = useCallback(
    async (keywords: string) => {
      await fetchDictTypes({ keywords, pageNum: 1 });
    },
    [fetchDictTypes]
  );

  // 重置字典类型查询
  const resetDictTypeQuery = useCallback(() => {
    const defaultParams: DictTypeQuery = { pageNum: 1, pageSize: 10 };
    setDictTypeQueryParams(defaultParams);
    fetchDictTypes(defaultParams);
  }, [fetchDictTypes]);

  // 获取字典数据分页列表
  const fetchDictItems = useCallback(
    async (params?: Partial<DictQuery>) => {
      setDictItemLoading(true);
      try {
        const { DictAPI } = await import("dehaze-sdk-js");
        const query = { ...dictItemQueryParams, ...params };
        const response = await DictAPI.getDictPage(query);
        setDictItems(response.list);
        setDictItemTotal(response.total);
        setDictItemQueryParams(query);
        return response;
      } catch (error) {
        console.error("获取字典数据列表失败:", error);
        Taro.showToast({ title: "获取字典数据列表失败", icon: "none" });
        throw error;
      } finally {
        setDictItemLoading(false);
      }
    },
    [dictItemQueryParams]
  );

  // 新增字典数据
  const createDictItem = useCallback(async (data: DictForm) => {
    try {
      const { DictAPI } = await import("dehaze-sdk-js");
      await DictAPI.addDict(data);
      Taro.showToast({ title: "新增字典数据成功", icon: "none" });
      return true;
    } catch (error) {
      console.error("新增字典数据失败:", error);
      Taro.showToast({ title: "新增字典数据失败", icon: "none" });
      throw error;
    }
  }, []);

  // 获取字典数据表单数据
  const fetchDictItemForm = useCallback(async (id: number) => {
    try {
      const { DictAPI } = await import("dehaze-sdk-js");
      return await DictAPI.getDictFormData(id);
    } catch (error) {
      console.error("获取字典数据表单失败:", error);
      Taro.showToast({ title: "获取字典数据表单失败", icon: "none" });
      throw error;
    }
  }, []);

  // 修改字典数据
  const updateDictItem = useCallback(async (id: number, data: DictForm) => {
    try {
      const { DictAPI } = await import("dehaze-sdk-js");
      await DictAPI.updateDict(id, data);
      Taro.showToast({ title: "修改字典数据成功", icon: "none" });
      return true;
    } catch (error) {
      console.error("修改字典数据失败:", error);
      Taro.showToast({ title: "修改字典数据失败", icon: "none" });
      throw error;
    }
  }, []);

  // 删除字典数据
  const deleteDictItems = useCallback(async (ids: string) => {
    try {
      const { DictAPI } = await import("dehaze-sdk-js");
      await DictAPI.deleteDictByIds(ids);
      Taro.showToast({ title: "删除字典数据成功", icon: "none" });
      return true;
    } catch (error) {
      console.error("删除字典数据失败:", error);
      Taro.showToast({ title: "删除字典数据失败", icon: "none" });
      throw error;
    }
  }, []);

  // 获取字典下拉选项
  const getDictOptions = useCallback(async (typeCode: string) => {
    try {
      const { DictAPI } = await import("dehaze-sdk-js");
      const options: OptionType[] = await DictAPI.getDictOptions(typeCode);
      return options;
    } catch (error) {
      console.error("获取字典选项失败:", error);
      return [];
    }
  }, []);

  return {
    // 字典类型状态
    dictTypes,
    dictTypeLoading,
    dictTypeTotal,
    dictTypeQueryParams,

    // 字典数据状态
    dictItems,
    dictItemLoading,
    dictItemTotal,
    dictItemQueryParams,

    // 字典类型操作
    fetchDictTypes,
    createDictType,
    fetchDictTypeForm,
    updateDictType,
    deleteDictTypes,
    searchDictTypes,
    resetDictTypeQuery,

    // 字典数据操作
    fetchDictItems,
    createDictItem,
    fetchDictItemForm,
    updateDictItem,
    deleteDictItems,
    getDictOptions,

    setDictTypeQueryParams,
    setDictItemQueryParams,
  };
};
