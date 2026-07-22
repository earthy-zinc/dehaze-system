import { useState, useCallback } from "react";
import Taro from "@tarojs/taro";
import type {
  RoleQuery,
  RolePageVO,
  RoleForm,
  OptionType,
} from "dehaze-sdk-js";

export const useRoleManagement = () => {
  // 状态定义
  const [roles, setRoles] = useState<RolePageVO[]>([]);
  const [permissions, setPermissions] = useState<OptionType[]>([]);
  const [loading, setLoading] = useState(false);
  const [total, setTotal] = useState(0);
  const [queryParams, setQueryParams] = useState<RoleQuery>({
    pageNum: 1,
    pageSize: 10,
  });

  // 获取角色列表
  const fetchRoles = useCallback(
    async (params?: Partial<RoleQuery>) => {
      setLoading(true);
      try {
        const { RoleAPI } = await import("dehaze-sdk-js");
        const query = { ...queryParams, ...params };
        const response = await RoleAPI.getPage(query);
        setRoles(response.list);
        setTotal(response.total);
        setQueryParams(query);
        return response;
      } catch (error) {
        console.error("获取角色列表失败:", error);
        Taro.showToast({ title: "获取角色列表失败", icon: "none" });
        throw error;
      } finally {
        setLoading(false);
      }
    },
    [queryParams]
  );

  // 获取权限树
  const fetchPermissions = useCallback(async () => {
    try {
      const { MenuAPI } = await import("dehaze-sdk-js");
      const menuList = await MenuAPI.getOptions();
      setPermissions(menuList);
      return menuList;
    } catch (error) {
      console.error("获取权限列表失败:", error);
      Taro.showToast({ title: "获取权限列表失败", icon: "none" });
      throw error;
    }
  }, []);

  // 获取角色权限
  const fetchRolePermissions = useCallback(async (roleId: number) => {
    try {
      const { RoleAPI } = await import("dehaze-sdk-js");
      const permissionIds = await RoleAPI.getRoleMenuIds(roleId);
      return permissionIds;
    } catch (error) {
      console.error("获取角色权限失败:", error);
      Taro.showToast({ title: "获取角色权限失败", icon: "none" });
      throw error;
    }
  }, []);

  // 创建角色
  const createRole = useCallback(
    async (roleData: RoleForm) => {
      try {
        const { RoleAPI } = await import("dehaze-sdk-js");
        await RoleAPI.add(roleData);
        await fetchRoles();
        Taro.showToast({ title: "创建角色成功", icon: "none" });
        return true;
      } catch (error) {
        console.error("创建角色失败:", error);
        Taro.showToast({ title: "创建角色失败", icon: "none" });
        throw error;
      }
    },
    [fetchRoles]
  );

  // 更新角色
  const updateRole = useCallback(
    async (id: number, roleData: RoleForm) => {
      try {
        const { RoleAPI } = await import("dehaze-sdk-js");
        await RoleAPI.update(id, roleData);
        await fetchRoles();
        Taro.showToast({ title: "更新角色成功", icon: "none" });
        return true;
      } catch (error) {
        console.error("更新角色失败:", error);
        Taro.showToast({ title: "更新角色失败", icon: "none" });
        throw error;
      }
    },
    [fetchRoles]
  );

  // 删除角色
  const deleteRole = useCallback(
    async (ids: string | number) => {
      try {
        const { RoleAPI } = await import("dehaze-sdk-js");
        await RoleAPI.deleteByIds(String(ids));
        await fetchRoles();
        Taro.showToast({ title: "删除角色成功", icon: "none" });
        return true;
      } catch (error) {
        console.error("删除角色失败:", error);
        Taro.showToast({ title: "删除角色失败", icon: "none" });
        throw error;
      }
    },
    [fetchRoles]
  );

  // 分配权限
  const assignPermissions = useCallback(
    async (roleId: number, permissionIds: number[]) => {
      try {
        const { RoleAPI } = await import("dehaze-sdk-js");
        await RoleAPI.updateRoleMenus(roleId, permissionIds);
        // Toast 将在组件中处理
        return true;
      } catch (error) {
        console.error("分配权限失败:", error);
        // Toast 将在组件中处理
        throw error;
      }
    },
    []
  );

  // 获取角色选项列表
  const getRoleOptions = useCallback(async () => {
    try {
      const { RoleAPI } = await import("dehaze-sdk-js");
      const options = await RoleAPI.getOptions();
      return options;
    } catch (error) {
      console.error("获取角色选项失败:", error);
      Taro.showToast({ title: "获取角色选项失败", icon: "none" });
      throw error;
    }
  }, []);

  // 搜索角色
  const searchRoles = useCallback(
    async (keywords: string) => {
      await fetchRoles({ keywords, pageNum: 1 });
    },
    [fetchRoles]
  );

  // 重置查询参数
  const resetQuery = useCallback(() => {
    const defaultParams = {
      pageNum: 1,
      pageSize: 10,
    };
    setQueryParams(defaultParams);
    fetchRoles(defaultParams);
  }, [fetchRoles]);

  return {
    // 状态
    roles,
    permissions,
    loading,
    total,
    queryParams,

    // 操作方法
    fetchRoles,
    fetchPermissions,
    fetchRolePermissions,
    createRole,
    updateRole,
    deleteRole,
    assignPermissions,
    getRoleOptions,
    searchRoles,
    resetQuery,
    setQueryParams,
  };
};
