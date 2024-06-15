import { useDeepCompareEffect, useLatest } from "ahooks";
import { BasicTarget, getTargetElement } from "ahooks/es/utils/domTarget";
import { useCallback, useRef } from "react";

type UseResizeObserver = (
  /**
   * @zh dom元素
   * @en dom element
   */
  target: BasicTarget,
  /**
   * @zh 回调
   * @en callback
   */
  callback: ResizeObserverCallback,
  /**
   * @zh `resizeObserver` 参数
   * @en options passed to `resizeObserver`
   */
  options?: ResizeObserverOptions
) => () => void;

/**
 * 使用 ResizeObserver 跟踪元素大小。
 */
export const useResizeObserver: UseResizeObserver = (
  target,
  callback: ResizeObserverCallback,
  options: ResizeObserverOptions = {}
): (() => void) => {
  const savedCallback = useLatest(callback);
  const observerRef = useRef<ResizeObserver>();

  const stop = useCallback(() => {
    if (observerRef.current) {
      observerRef.current.disconnect();
    }
  }, []);
  useDeepCompareEffect(() => {
    const element = getTargetElement(target);
    if (!element) {
      return;
    }
    observerRef.current = new ResizeObserver(savedCallback.current);
    observerRef.current.observe(element, options);

    return stop;
  }, [savedCallback, stop, target, options]);

  return stop;
};
