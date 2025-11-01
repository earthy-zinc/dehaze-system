import { useState, useCallback } from 'react';
import { Toast } from '@taroify/core';
import type { UserQuery, UserPageVO, UserForm } from 'dehaze-sdk-js';

export const useUserManagement = () => {
  // 状态定义
  const [users, setUsers] = useState<UserPageVO[]>([]);
  const [loading, setLoading] = useState(false);
  const [total, setTotal] = useState(0);
  const [queryParams, setQueryParams] = useState<UserQuery>({
    pageNum: 1,
    pageSize: 10,
  });

  // 获取用户列表
  const fetchUsers = useCallback(async (params?: Partial<UserQuery>) => {
    setLoading(true);
    try {
      const { UserAPI } = await import('dehaze-sdk-js');
      const query = { ...queryParams, ...params };
      const response = await UserAPI.getPage(query);
      setUsers(response.list);
      setTotal(response.total);
      setQueryParams(query);
      return response;
    } catch (error) {
      console.error('获取用户列表失败:', error);
      Toast.open({ message: '获取用户列表失败', position: 'top' });
      throw error;
    } finally {
      setLoading(false);
    }
  }, [queryParams]);

  // 创建用户
  const createUser = useCallback(async (userData: UserForm) => {
    try {
      const { UserAPI } = await import('dehaze-sdk-js');
      await UserAPI.add(userData);
      await fetchUsers(); // 重新获取列表
      Toast.open({ message: '创建用户成功', position: 'top' });
      return true;
    } catch (error) {
      console.error('创建用户失败:', error);
      Toast.open({ message: '创建用户失败', position: 'top' });
      throw error;
    }
  }, [fetchUsers]);

  // 更新用户
  const updateUser = useCallback(async (id: number, userData: UserForm) => {
    try {
      const { UserAPI } = await import('dehaze-sdk-js');
      await UserAPI.update(id, userData);
      await fetchUsers(); // 重新获取列表
      Toast.open({ message: '更新用户成功', position: 'top' });
      return true;
    } catch (error) {
      console.error('更新用户失败:', error);
      Toast.open({ message: '更新用户失败', position: 'top' });
      throw error;
    }
  }, [fetchUsers]);

  // 删除用户
  const deleteUser = useCallback(async (ids: string | number) => {
    try {
      const { UserAPI } = await import('dehaze-sdk-js');
      await UserAPI.deleteByIds(String(ids));
      await fetchUsers(); // 重新获取列表
      Toast.open({ message: '删除用户成功', position: 'top' });
      return true;
    } catch (error) {
      console.error('删除用户失败:', error);
      Toast.open({ message: '删除用户失败', position: 'top' });
      throw error;
    }
  }, [fetchUsers]);

  // 重置密码
  const resetPassword = useCallback(async (userId: number, newPassword: string) => {
    try {
      const { UserAPI } = await import('dehaze-sdk-js');
      await UserAPI.updatePassword(userId, newPassword);
      Toast.open({ message: '密码重置成功', position: 'top' });
      return true;
    } catch (error) {
      console.error('重置密码失败:', error);
      Toast.open({ message: '重置密码失败', position: 'top' });
      throw error;
    }
  }, []);

  // 获取用户详情
  const getUserDetail = useCallback(async (id: number) => {
    try {
      const { UserAPI } = await import('dehaze-sdk-js');
      const userData = await UserAPI.getFormData(id);
      return userData;
    } catch (error) {
      console.error('获取用户详情失败:', error);
      Toast.open({ message: '获取用户详情失败', position: 'top' });
      throw error;
    }
  }, []);

  // 搜索用户
  const searchUsers = useCallback(async (keywords: string) => {
    await fetchUsers({ keywords, pageNum: 1 });
  }, [fetchUsers]);

  // 重置查询参数
  const resetQuery = useCallback(() => {
    const defaultParams = {
      pageNum: 1,
      pageSize: 10,
    };
    setQueryParams(defaultParams);
    fetchUsers(defaultParams);
  }, [fetchUsers]);

  return {
    // 状态
    users,
    loading,
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
    resetQuery,
    setQueryParams,
  };
};