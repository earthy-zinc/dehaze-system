import { useState, useCallback } from "react";

/**
 * 通用分页 Hook（Taro）
 * @param defaultPageSize 默认每页条数
 */
export function usePagination(defaultPageSize = 10) {
  const [pageNum, setPageNum] = useState(1);
  const [pageSize, setPageSize] = useState(defaultPageSize);
  const [total, setTotal] = useState(0);

  const handlePageChange = useCallback((page: number) => setPageNum(page), []);
  const handleSizeChange = useCallback((size: number) => {
    setPageSize(size);
    setPageNum(1);
  }, []);
  const reset = useCallback(() => setPageNum(1), []);
  const setTotalCount = useCallback((n: number) => setTotal(n), []);

  return {
    pageNum,
    pageSize,
    total,
    handlePageChange,
    handleSizeChange,
    reset,
    setTotal: setTotalCount,
  };
}
