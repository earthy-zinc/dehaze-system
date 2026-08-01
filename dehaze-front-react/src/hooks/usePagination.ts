import { useMemo, useState } from "react";

export interface UsePaginationOptions {
  defaultPageSize?: number;
}

export interface UsePaginationReturn {
  pageNum: number;
  pageSize: number;
  total: number;
  setPageNum: (n: number) => void;
  setPageSize: (n: number) => void;
  setTotal: (n: number) => void;
  reset: () => void;
  antdPagination: {
    current: number;
    pageSize: number;
    total: number;
    showSizeChanger: boolean;
    showQuickJumper: boolean;
    onChange: (page: number, size: number) => void;
  };
}

export function usePagination(
  options: UsePaginationOptions = {}
): UsePaginationReturn {
  const { defaultPageSize = 10 } = options;
  const [pageNum, setPageNum] = useState(1);
  const [pageSize, setPageSize] = useState(defaultPageSize);
  const [total, setTotal] = useState(0);

  const reset = () => {
    setPageNum(1);
    setPageSize(defaultPageSize);
    setTotal(0);
  };

  const antdPagination = useMemo(
    () => ({
      current: pageNum,
      pageSize,
      total,
      showSizeChanger: true,
      showQuickJumper: true,
      onChange: (page: number, size: number) => {
        setPageNum(page);
        setPageSize(size);
      },
    }),
    [pageNum, pageSize, total]
  );

  return {
    pageNum,
    pageSize,
    total,
    setPageNum,
    setPageSize,
    setTotal,
    reset,
    antdPagination,
  };
}
