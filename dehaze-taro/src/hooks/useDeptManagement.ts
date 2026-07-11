import { useState, useCallback } from 'react';
import { Toast } from '@taroify/core';
import type { OptionType } from 'dehaze-sdk-js';

export const useDeptManagement = () => {
  // 状态定义
  const [deptOptions, setDeptOptions] = useState<OptionType[]>([]);
  const [loading, setLoading] = useState(false);

  // 获取部门选项列表
  const fetchDeptOptions = useCallback(async () => {
    setLoading(true);
    try {
      const { DeptAPI } = await import('dehaze-sdk-js');
      const options = await DeptAPI.getOptions();
      setDeptOptions(options);
      return options;
    } catch (error) {
      console.error('获取部门选项失败:', error);
      Toast.open({ message: '获取部门选项失败', position: 'top' });
      throw error;
    } finally {
      setLoading(false);
    }
  }, []);

  // 获取部门列表（树形结构）
  const fetchDeptList = useCallback(async () => {
    setLoading(true);
    try {
      const { DeptAPI } = await import('dehaze-sdk-js');
      const deptList = await DeptAPI.getList();
      return deptList;
    } catch (error) {
      console.error('获取部门列表失败:', error);
      Toast.open({ message: '获取部门列表失败', position: 'top' });
      throw error;
    } finally {
      setLoading(false);
    }
  }, []);

  // 创建部门
  const createDept = useCallback(async (deptData: any) => {
    try {
      const { DeptAPI } = await import('dehaze-sdk-js');
      await DeptAPI.add(deptData);
      await fetchDeptOptions();
      Toast.open({ message: '创建部门成功', position: 'top' });
      return true;
    } catch (error) {
      console.error('创建部门失败:', error);
      Toast.open({ message: '创建部门失败', position: 'top' });
      throw error;
    }
  }, [fetchDeptOptions]);

  // 更新部门
  const updateDept = useCallback(async (id: number, deptData: any) => {
    try {
      const { DeptAPI } = await import('dehaze-sdk-js');
      await DeptAPI.update(id, deptData);
      await fetchDeptOptions();
      Toast.open({ message: '更新部门成功', position: 'top' });
      return true;
    } catch (error) {
      console.error('更新部门失败:', error);
      Toast.open({ message: '更新部门失败', position: 'top' });
      throw error;
    }
  }, [fetchDeptOptions]);

  // 删除部门
  const deleteDept = useCallback(async (ids: string | number) => {
    try {
      const { DeptAPI } = await import('dehaze-sdk-js');
      const id = Number(ids);
      if (ids.toString().includes(',')) {
        const idArray = ids.toString().split(',').map(Number);
        await DeptAPI.batchDelete(idArray);
      } else {
        await DeptAPI.deleteById(id);
      }
      await fetchDeptOptions();
      Toast.open({ message: '删除部门成功', position: 'top' });
      return true;
    } catch (error) {
      console.error('删除部门失败:', error);
      Toast.open({ message: '删除部门失败', position: 'top' });
      throw error;
    }
  }, [fetchDeptOptions]);

  return {
    // 状态
    deptOptions,
    loading,

    // 操作方法
    fetchDeptOptions,
    fetchDeptList,
    createDept,
    updateDept,
    deleteDept,
  };
};