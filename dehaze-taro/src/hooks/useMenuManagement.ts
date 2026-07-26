import { useState, useCallback } from "react";
import Taro from "@tarojs/taro";
import type { MenuVO, MenuForm, MenuQuery, OptionType } from "dehaze-sdk-js";

export const useMenuManagement = () => {
  // 菜单树形列表
  const [menuList, setMenuList] = useState<MenuVO[]>([]);
  const [loading, setLoading] = useState(false);
  // 上级菜单下拉选项
  const [menuOptions, setMenuOptions] = useState<OptionType[]>([]);
  const [queryParams, setQueryParams] = useState<MenuQuery>({});

  // 获取菜单树形列表
  const fetchMenus = useCallback(
    async (params?: Partial<MenuQuery>) => {
      setLoading(true);
      try {
        const { MenuAPI } = await import("dehaze-sdk-js");
        const query = { ...queryParams, ...params };
        const list = await MenuAPI.getList(query);
        setMenuList(list || []);
        setQueryParams(query);
        return list;
      } catch (error) {
        console.error("获取菜单列表失败:", error);
        Taro.showToast({ title: "获取菜单列表失败", icon: "none" });
        throw error;
      } finally {
        setLoading(false);
      }
    },
    [queryParams]
  );

  // 获取上级菜单下拉选项
  const fetchMenuOptions = useCallback(async () => {
    try {
      const { MenuAPI } = await import("dehaze-sdk-js");
      const options = await MenuAPI.getOptions();
      setMenuOptions(options || []);
      return options;
    } catch (error) {
      console.error("获取菜单下拉选项失败:", error);
      Taro.showToast({ title: "获取菜单下拉选项失败", icon: "none" });
      throw error;
    }
  }, []);

  // 获取菜单表单数据
  const fetchMenuForm = useCallback(async (id: number) => {
    try {
      const { MenuAPI } = await import("dehaze-sdk-js");
      return await MenuAPI.getFormData(id);
    } catch (error) {
      console.error("获取菜单表单数据失败:", error);
      Taro.showToast({ title: "获取菜单表单数据失败", icon: "none" });
      throw error;
    }
  }, []);

  // 新增菜单
  const createMenu = useCallback(
    async (data: MenuForm) => {
      try {
        const { MenuAPI } = await import("dehaze-sdk-js");
        await MenuAPI.add(data);
        await fetchMenus();
        Taro.showToast({ title: "新增菜单成功", icon: "none" });
        return true;
      } catch (error) {
        console.error("新增菜单失败:", error);
        Taro.showToast({ title: "新增菜单失败", icon: "none" });
        throw error;
      }
    },
    [fetchMenus]
  );

  // 修改菜单
  const updateMenu = useCallback(
    async (id: string, data: MenuForm) => {
      try {
        const { MenuAPI } = await import("dehaze-sdk-js");
        await MenuAPI.update(id, data);
        await fetchMenus();
        Taro.showToast({ title: "修改菜单成功", icon: "none" });
        return true;
      } catch (error) {
        console.error("修改菜单失败:", error);
        Taro.showToast({ title: "修改菜单失败", icon: "none" });
        throw error;
      }
    },
    [fetchMenus]
  );

  // 删除菜单
  const deleteMenu = useCallback(
    async (id: number) => {
      try {
        const { MenuAPI } = await import("dehaze-sdk-js");
        await MenuAPI.deleteByIds(String(id));
        await fetchMenus();
        Taro.showToast({ title: "删除菜单成功", icon: "none" });
        return true;
      } catch (error) {
        console.error("删除菜单失败:", error);
        Taro.showToast({ title: "删除菜单失败", icon: "none" });
        throw error;
      }
    },
    [fetchMenus]
  );

  // 搜索菜单
  const searchMenus = useCallback(
    async (keywords: string) => {
      await fetchMenus({ keywords });
    },
    [fetchMenus]
  );

  // 重置查询
  const resetQuery = useCallback(() => {
    const defaultParams: MenuQuery = {};
    setQueryParams(defaultParams);
    fetchMenus(defaultParams);
  }, [fetchMenus]);

  return {
    // 状态
    menuList,
    loading,
    menuOptions,
    queryParams,

    // 操作方法
    fetchMenus,
    fetchMenuOptions,
    fetchMenuForm,
    createMenu,
    updateMenu,
    deleteMenu,
    searchMenus,
    resetQuery,
    setQueryParams,
  };
};
