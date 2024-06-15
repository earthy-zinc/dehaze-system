import { WaterfallProps } from "@/components/Waterfall/types.ts";
import { addClass, hasClass, prefixStyle } from "@/utils";
import { CssStyleObject } from "@/utils/types.ts";
import React, { useCallback, useEffect, useState } from "react";

const transform = prefixStyle("transform");
const duration = prefixStyle("animation-duration");
const delay = prefixStyle("animation-delay");
const transition = prefixStyle("transition");
const fillMode = prefixStyle("animation-fill-mode");

// 动画
function addAnimation(props: WaterfallProps) {
  return (item: HTMLElement, callback?: () => void) => {
    const content = item!.firstChild as HTMLElement;
    if (content && !hasClass(content, props.animationPrefix)) {
      const durationSec = `${props.animationDuration / 1000}s`;
      const delaySec = `${props.animationDelay / 1000}s`;
      const style = content.style as CssStyleObject;
      addClass(content, props.animationPrefix);
      addClass(content, props.animationEffect);

      if (duration) style[duration] = durationSec;

      if (delay) style[delay] = delaySec;

      if (fillMode) style[fillMode] = "both";

      if (callback) {
        setTimeout(() => {
          callback();
        }, props.animationDuration + props.animationDelay);
      }
    }
  };
}

const useLayout = (
  props: WaterfallProps,
  colWidth: number,
  cols: number,
  offsetX: number,
  waterfallWrapper: React.MutableRefObject<HTMLElement>
) => {
  const [posY, setPosY] = useState(
    Array(cols).fill(props.hasAroundGutter ? props.gutter : 0)
  );
  const [wrapperHeight, setWrapperHeight] = useState(0);

  const getX = (index: number) => {
    const count = props.hasAroundGutter ? index + 1 : index;
    return props.gutter * count + colWidth * index + offsetX;
  };

  const initY = useCallback(() => {
    setPosY(new Array(cols).fill(props.hasAroundGutter ? props.gutter : 0));
  }, [cols, props.gutter, props.hasAroundGutter]);

  const animation = addAnimation(props);

  // 排版
  const layoutHandle = async (): Promise<boolean> => {
    return new Promise((resolve) => {
      // 初始化y集合
      initY();

      // 构造列表
      const items: HTMLElement[] = [];
      if (waterfallWrapper) {
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
        posY[minYIndex] += height + props.gutter;

        // 添加入场动画
        animation(curItem, () => {
          // 添加动画时间
          const time = props.posDuration / 1000;
          if (transition) style[transition] = `transform ${time}s`;
        });
      }

      setWrapperHeight(Math.max.apply(null, posY));

      setTimeout(() => {
        resolve(true);
      }, props.posDuration);
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
