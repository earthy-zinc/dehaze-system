import type { Ref } from "vue";
import { ref } from "vue";

/**
 * 分页参数响应式状态类型。
 */
export interface PaginationParams {
  pageNum: Ref<number>;
  pageSize: Ref<number>;
  total: Ref<number>;
}

/**
 * 分页逻辑配置选项。
 */
export interface UsePaginationOptions {
  /**
   * 初始页码，默认 1。
   */
  initialPage?: number;
  /**
   * 初始每页条数，默认 20。
   */
  initialPageSize?: number;
  /**
   * 总条数，可在外部通过返回的 total ref 更新。
   */
  initialTotal?: number;
}

/**
 * 通用分页 Composable，封装分页常见操作。
 *
 * @example
 * ```ts
 * const { pageNum, pageSize, total, handlePageChange, handleSizeChange, reset } = usePagination();
 *
 * const fetchData = () => {
 *   // 使用 pageNum.value, pageSize.value, total.value 发起 API 请求
 * };
 *
 * watch([pageNum, pageSize], () => {
 *   fetchData();
 * });
 * ```
 *
 * @param options - 分页配置选项
 * @returns 分页状态与操作方法
 */
export function usePagination(options: UsePaginationOptions = {}) {
  const pageNum = ref(options.initialPage ?? 1);
  const pageSize = ref(options.initialPageSize ?? 20);
  const total = ref(options.initialTotal ?? 0);

  /**
   * 切换页码。
   * @param newPage - 目标页码
   */
  const handlePageChange = (newPage: number) => {
    pageNum.value = Math.max(1, newPage);
  };

  /**
   * 切换每页条数。
   * @param newSize - 新的每页条数
   */
  const handleSizeChange = (newSize: number) => {
    pageSize.value = newSize;
    pageNum.value = 1;
  };

  /**
   * 重置到第一页。
   */
  const reset = () => {
    pageNum.value = 1;
    total.value = 0;
  };

  return {
    pageNum,
    pageSize,
    total,
    handlePageChange,
    handleSizeChange,
    reset,
  };
}
