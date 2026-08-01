import { ref, watch, type Ref } from "vue";

/**
 * 防抖函数包装器。
 *
 * @param fn - 需要防抖的回调函数
 * @param delay - 防抖延迟（毫秒），默认 300
 * @returns 防抖后的函数
 *
 * @example
 * ```ts
 * const search = (keyword: string) => console.log('search:', keyword);
 * const debouncedSearch = useDebounce(search, 500);
 *
 * // 每次调用都会重置计时器，只有停止触发 500ms 后才执行
 * debouncedSearch('hello');
 * debouncedSearch('world'); // 前一次被取消
 * ```
 */
export function useDebounce<T extends (...args: any[]) => void>(
  fn: T,
  delay: number = 300
): (...args: Parameters<T>) => void {
  let timer: ReturnType<typeof setTimeout> | null = null;

  return function (this: any, ...args: Parameters<T>) {
    if (timer) {
      clearTimeout(timer);
    }
    timer = setTimeout(() => {
      fn.apply(this, args);
    }, delay);
  };
}

/**
 * 防抖的 Ref，适用于搜索框等 v-model 绑定场景。
 *
 * @param initialValue - 初始值
 * @param delay - 防抖延迟（毫秒），默认 300
 * @returns 防抖后的 ref，其 value 在停止赋值 delay 毫秒后更新
 *
 * @example
 * ```ts
 * const keyword = useDebouncedRef('', 300);
 *
 * // 模板中直接绑定
 * // <el-input v-model="keyword" placeholder="搜索..." />
 *
 * // watch keyword.value 即可在输入停止后触发搜索
 * watch(keyword, (newVal) => {
 *   if (newVal.trim()) fetchResults(newVal);
 * });
 * ```
 */
export function useDebouncedRef<T>(
  initialValue: T,
  delay: number = 300
): Ref<T> {
  const internalRef = ref(initialValue);
  const debouncedRef = ref(initialValue) as Ref<T>;

  let timer: ReturnType<typeof setTimeout> | null = null;

  watch(internalRef, (newValue) => {
    if (timer) {
      clearTimeout(timer);
    }
    timer = setTimeout(() => {
      debouncedRef.value = newValue;
    }, delay);
  });

  // 暴露内部 ref 用于 v-model 双向绑定
  Object.defineProperty(debouncedRef, "internal", {
    get: () => internalRef,
    enumerable: false,
    configurable: true,
  });

  return debouncedRef as Ref<T> & { internal: Ref<T> };
}
