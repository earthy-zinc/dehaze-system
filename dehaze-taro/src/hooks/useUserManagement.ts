import { useState, useCallback } from "react";
import Taro from "@tarojs/taro";
import type { UserQuery, UserPageVO, UserForm } from "dehaze-sdk-js";
import { getErrorMessage } from "@/utils/error";

export const useUserManagement = () => {
  // 状态定义
  const [users, setUsers] = useState<UserPageVO[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [total, setTotal] = useState(0);
  const [queryParams, setQueryParams] = useState<UserQuery>({
    pageNum: 1,
    pageSize: 10,
  });

  // 获取用户列表
  const fetchUsers = useCallback(
    async (params?: Partial<UserQuery>) => {
      setLoading(true);
      setLoadError(null);
      try {
        const { UserAPI } = await import("dehaze-sdk-js");
        const query = { ...queryParams, ...params };
        const response = await UserAPI.getPage(query);
        setUsers(response.list);
        setTotal(response.total);
        setQueryParams(query);
        return response;
      } catch (error: unknown) {
        setLoadError(getErrorMessage(error, "获取用户列表失败，请重试"));
        throw error;
      } finally {
        setLoading(false);
      }
    },
    [queryParams]
  );

  // 创建用户
  const createUser = useCallback(
    async (userData: UserForm) => {
      try {
        const { UserAPI } = await import("dehaze-sdk-js");
        await UserAPI.add(userData);
        await fetchUsers(); // 重新获取列表
        Taro.showToast({ title: "创建用户成功", icon: "none" });
        return true;
      } catch (error) {
        console.error("创建用户失败:", error);
        Taro.showToast({ title: "创建用户失败", icon: "none" });
        throw error;
      }
    },
    [fetchUsers]
  );

  // 更新用户
  const updateUser = useCallback(
    async (id: number, userData: UserForm) => {
      try {
        const { UserAPI } = await import("dehaze-sdk-js");
        await UserAPI.update(id, userData);
        await fetchUsers(); // 重新获取列表
        Taro.showToast({ title: "更新用户成功", icon: "none" });
        return true;
      } catch (error) {
        console.error("更新用户失败:", error);
        Taro.showToast({ title: "更新用户失败", icon: "none" });
        throw error;
      }
    },
    [fetchUsers]
  );

  // 删除用户
  const deleteUser = useCallback(
    async (ids: string | number) => {
      try {
        const { UserAPI } = await import("dehaze-sdk-js");
        await UserAPI.deleteByIds(String(ids));
        await fetchUsers(); // 重新获取列表
        Taro.showToast({ title: "删除用户成功", icon: "none" });
        return true;
      } catch (error) {
        console.error("删除用户失败:", error);
        Taro.showToast({ title: "删除用户失败", icon: "none" });
        throw error;
      }
    },
    [fetchUsers]
  );

  // 重置密码
  const resetPassword = useCallback(
    async (userId: number, newPassword: string) => {
      try {
        const { UserAPI } = await import("dehaze-sdk-js");
        await UserAPI.updatePassword(userId, newPassword);
        Taro.showToast({ title: "密码重置成功", icon: "none" });
        return true;
      } catch (error) {
        console.error("重置密码失败:", error);
        Taro.showToast({ title: "重置密码失败", icon: "none" });
        throw error;
      }
    },
    []
  );

  // 获取用户详情
  const getUserDetail = useCallback(async (id: number) => {
    try {
      const { UserAPI } = await import("dehaze-sdk-js");
      const userData = await UserAPI.getFormData(id);
      return userData;
    } catch (error) {
      console.error("获取用户详情失败:", error);
      Taro.showToast({ title: "获取用户详情失败", icon: "none" });
      throw error;
    }
  }, []);

  // 搜索用户
  const searchUsers = useCallback(
    async (keywords: string) => {
      await fetchUsers({ keywords, pageNum: 1 });
    },
    [fetchUsers]
  );

  return {
    // 状态
    users,
    loading,
    loadError,
    total,
    queryParams,

    // 操作方法
    fetchUsers,
    createUser,
    updateUser,
    deleteUser,
    resetPassword,
    getUserDetail,
    searchUsers,
    setQueryParams,
  };
};
