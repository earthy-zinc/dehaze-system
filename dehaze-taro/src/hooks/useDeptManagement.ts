import { useState, useCallback } from 'react';
import { Toast } from '@taroify/core';
import type { DeptVO, DeptForm, DeptQuery, OptionType } from 'dehaze-sdk-js';

export const useDeptManagement = () => {
  // 部门树形列表
  const [deptList, setDeptList] = useState<DeptVO[]>([]);
  // 上级部门下拉选项
  const [deptOptions, setDeptOptions] = useState<OptionType[]>([]);
  const [loading, setLoading] = useState(false);
  const [queryParams, setQueryParams] = useState<DeptQuery>({});

  // 获取部门树形列表
  const fetchDeptList = useCallback(async (params?: Partial<DeptQuery>) => {
    setLoading(true);
    try {
      const { DeptAPI } = await import('dehaze-sdk-js');
      const query = { ...queryParams, ...params };
      const list = await DeptAPI.getList(query);
      setDeptList(list || []);
      setQueryParams(query);
      return list;
    } catch (error) {
      console.error('获取部门列表失败:', error);
      Toast.open({ message: '获取部门列表失败', position: 'top' });
      throw error;
    } finally {
      setLoading(false);
    }
  }, [queryParams]);

  // 获取部门下拉选项
  const fetchDeptOptions = useCallback(async () => {
    try {
      const { DeptAPI } = await import('dehaze-sdk-js');
      const options = await DeptAPI.getOptions();
      setDeptOptions(options || []);
      return options;
    } catch (error) {
      console.error('获取部门选项失败:', error);
      Toast.open({ message: '获取部门选项失败', position: 'top' });
      throw error;
    }
  }, []);

  // 获取部门表单数据
  const fetchDeptForm = useCallback(async (id: number) => {
    try {
      const { DeptAPI } = await import('dehaze-sdk-js');
      return await DeptAPI.getFormData(id);
    } catch (error) {
      console.error('获取部门表单数据失败:', error);
      Toast.open({ message: '获取部门表单数据失败', position: 'top' });
      throw error;
    }
  }, []);

  // 新增部门
  const createDept = useCallback(async (data: DeptForm) => {
    try {
      const { DeptAPI } = await import('dehaze-sdk-js');
      await DeptAPI.add(data);
      await fetchDeptList();
      Toast.open({ message: '新增部门成功', position: 'top' });
      return true;
    } catch (error) {
      console.error('新增部门失败:', error);
      Toast.open({ message: '新增部门失败', position: 'top' });
      throw error;
    }
  }, [fetchDeptList]);

  // 修改部门
  const updateDept = useCallback(async (id: number, data: DeptForm) => {
    try {
      const { DeptAPI } = await import('dehaze-sdk-js');
      await DeptAPI.update(id, data);
      await fetchDeptList();
      Toast.open({ message: '修改部门成功', position: 'top' });
      return true;
    } catch (error) {
      console.error('修改部门失败:', error);
      Toast.open({ message: '修改部门失败', position: 'top' });
      throw error;
    }
  }, [fetchDeptList]);

  // 删除部门
  const deleteDept = useCallback(async (id: number) => {
    try {
      const { DeptAPI } = await import('dehaze-sdk-js');
      await DeptAPI.deleteById(id);
      await fetchDeptList();
      Toast.open({ message: '删除部门成功', position: 'top' });
      return true;
    } catch (error) {
      console.error('删除部门失败:', error);
      Toast.open({ message: '删除部门失败', position: 'top' });
      throw error;
    }
  }, [fetchDeptList]);

  // 搜索部门
  const searchDepts = useCallback(async (keywords: string) => {
    await fetchDeptList({ keywords });
  }, [fetchDeptList]);

  // 重置查询
  const resetQuery = useCallback(() => {
    const defaultParams: DeptQuery = {};
    setQueryParams(defaultParams);
    fetchDeptList(defaultParams);
  }, [fetchDeptList]);

  return {
    // 状态
    deptList,
    deptOptions,
    loading,
    queryParams,

    // 操作方法
    fetchDeptList,
    fetchDeptOptions,
    fetchDeptForm,
    createDept,
    updateDept,
    deleteDept,
    searchDepts,
    resetQuery,
    setQueryParams,
  };
};
