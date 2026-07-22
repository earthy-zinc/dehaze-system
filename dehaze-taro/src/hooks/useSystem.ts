import { useCallback } from "react";
import { useGlobalContext } from "@/stores/global";
import { storage } from "@/utils/storage";

export const useSystem = () => {
  const { state, dispatch } = useGlobalContext();

  // 设置选中的部门
  const setSelectedDept = useCallback(
    async (deptId: number | null) => {
      dispatch({
        type: "SET_SELECTED_DEPT",
        payload: deptId,
      });

      // 持久化存储
      await storage.setSelectedDept(deptId);
    },
    [dispatch]
  );

  // 清除缓存
  const clearCache = useCallback(() => {
    dispatch({ type: "CLEAR_CACHE" });
  }, [dispatch]);

  // 设置缓存过期时间
  const setCacheExpire = useCallback(
    (key: string, expireTime: number) => {
      dispatch({
        type: "SET_CACHE_EXPIRE",
        payload: { key, expireTime },
      });
    },
    [dispatch]
  );

  return {
    // 系统状态
    selectedDeptId: state.system.selectedDeptId,
    cacheExpireTime: state.system.cacheExpireTime,

    // 系统方法
    setSelectedDept,
    clearCache,
    setCacheExpire,
  };
};
