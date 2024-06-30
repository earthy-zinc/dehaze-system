import { addClass, hasClass, prefixStyle } from "@/utils";
import { CssStyleObject, Nullable } from "@/utils/types";
import React, { useCallback, useEffect, useState } from "react";

const transform = prefixStyle("transform");
const duration = prefixStyle("animation-duration");
const delay = prefixStyle("animation-delay");
const transition = prefixStyle("transition");
const fillMode = prefixStyle("animation-fill-mode");

// 动画
function addAnimation(
  animationPrefix: string,
  animationEffect: string,
  animationDelay: number,
  animationDuration: number
) {
  return (item: HTMLElement, callback?: () => void) => {
    const content = item!.firstChild as HTMLElement;
    if (content && !hasClass(content, animationPrefix)) {
      const durationSec = `${animationDuration / 1000}s`;
      const delaySec = `${animationDelay / 1000}s`;
      const style = content.style as CssStyleObject;
      addClass(content, animationPrefix);
      addClass(content, animationEffect);

      if (duration) style[duration] = durationSec;

      if (delay) style[delay] = delaySec;

      if (fillMode) style[fillMode] = "both";

      if (callback) {
        setTimeout(() => {
          callback();
        }, animationDuration + animationDelay);
      }
    }
  };
}

const useLayout = (
  hasAroundGutter: boolean,
  gutter: number,
  posDuration: number,
  animationPrefix: string,
  animationEffect: string,
  animationDelay: number,
  animationDuration: number,
  colWidth: number,
  cols: number,
  offsetX: number,
  waterfallWrapper: React.MutableRefObject<Nullable<HTMLDivElement>>
) => {
  const [posY, setPosY] = useState(
    Array(cols).fill(hasAroundGutter ? gutter : 0)
  );
  const [wrapperHeight, setWrapperHeight] = useState<number>(0);

  const getX = (index: number) => {
    const count = hasAroundGutter ? index + 1 : index;
    return gutter * count + colWidth * index + offsetX;
  };

  const initY = useCallback(() => {
    setPosY(new Array(cols).fill(hasAroundGutter ? gutter : 0));
  }, [cols, gutter, hasAroundGutter]);

  const animation = addAnimation(
    animationPrefix,
    animationEffect,
    animationDelay,
    animationDuration
  );

  // 排版
  const layoutHandle = async (): Promise<boolean> => {
    return new Promise((resolve) => {
      // 初始化y集合
      initY();

      // 构造列表
      const items: HTMLElement[] = [];
      if (waterfallWrapper.current) {
        waterfallWrapper.current.childNodes.forEach((el: any) => {
          if (el!.className === "waterfall-item") items.push(el);
        });
      }

      // 获取节点
      if (items.length === 0) return false;

      // 遍历节点
      for (const element of items) {
        const curItem = element as HTMLElement;
        // 最小的y值
        const minY = Math.min.apply(null, posY);
        // 最小y的下标
        const minYIndex = posY.indexOf(minY);
        // 当前下标对应的x
        const curX = getX(minYIndex);

        // 设置x,y,width
        const style = curItem.style as CssStyleObject;

        // 设置偏移
        if (transform) style[transform] = `translate3d(${curX}px,${minY}px, 0)`;
        style.width = `${colWidth}px`;

        style.visibility = "visible";

        // 更新当前index的y值
        const { height } = curItem.getBoundingClientRect();
        posY[minYIndex] += height + gutter;

        // 添加入场动画
        animation(curItem, () => {
          // 添加动画时间
          const time = posDuration / 1000;
          if (transition) style[transition] = `transform ${time}s`;
        });
      }

      setWrapperHeight(Math.max.apply(null, posY));

      setTimeout(() => {
        resolve(true);
      }, posDuration);
    });
  };

  useEffect(() => {
    initY();
  }, [cols, initY]);

  return {
    wrapperHeight,
    layoutHandle,
  };
};

export default useLayout;
