/**
 * 布局相关 Hook
 */
import { useState, useEffect } from "react";
import Taro from "@tarojs/taro";

/**
 * 获取状态栏高度
 */
export function useStatusBarHeight(): number {
  const [height, setHeight] = useState(0);

  useEffect(() => {
    try {
      const sysInfo = Taro.getSystemInfoSync();
      setHeight(sysInfo.statusBarHeight || 0);
    } catch (error) {
      void error;
    }
  }, []);

  return height;
}
