import { ref, type Ref } from "vue";

/**
 * 异步任务结果类型。
 */
export interface AsyncTaskResult<T = unknown> {
  /** 是否正在执行 */
  loading: Ref<boolean>;
  /** 错误信息，执行失败时自动捕获 */
  error: Ref<Error | null>;
  /** 执行结果数据 */
  data: Ref<T | null>;
  /** 执行异步函数 */
  execute: (debounce?: number) => Promise<T | null>;
}

/**
 * 异步任务状态管理 Composable，封装 loading/error/data 三态。
 *
 * @param asyncFn - 异步函数，接收 debounce 延迟参数（可选）
 * @returns 任务状态与方法
 *
 * @example
 * ```ts
 * const { loading, error, data, execute } = useAsyncTask(async (debounce?: number) => {
 *   await sleep(debounce ?? 0);
 *   return await fetchSomeData();
 * });
 *
 * // 直接执行
 * execute();
 *
 * // 带防抖执行
 * execute(300);
 * ```
 */
export function useAsyncTask<T>(
  asyncFn: (debounce?: number) => Promise<T>
): AsyncTaskResult<T> {
  const loading = ref(false);
  const error = ref<Error | null>(null);
  const data = ref<T | null>(null);

  const execute = async (debounce?: number): Promise<T | null> => {
    loading.value = true;
    error.value = null;

    try {
      if (debounce) {
        await new Promise<void>((resolve) => {
          setTimeout(resolve, debounce);
        });
      }
      const result = await asyncFn(debounce);
      data.value = result;
      return result;
    } catch (e: unknown) {
      error.value = e instanceof Error ? e : new Error(String(e));
      return null;
    } finally {
      loading.value = false;
    }
  };

  return {
    loading,
    error,
    data: data as Ref<T | null>,
    execute,
  };
}
