import { addClass, hasClass, prefixStyle } from "@/utils";
import { CssStyleObject, Nullable } from "@/utils/types";
import { useResizeObserver } from "@vueuse/core";
import { computed, ref } from "vue";
import { ItemWidthProps, WaterfallProps } from "./types";

/**
 * @description: 获取当前窗口尺寸下格子的宽度
 */
const getItemWidth = ({
  breakpoints,
  wrapperWidth,
  gutter,
  hasAroundGutter,
  initWidth,
}: ItemWidthProps) => {
  // 获取升序尺寸集合
  const sizeList: number[] = Object.keys(breakpoints)
    .map((key) => {
      return Number(key);
    })
    .sort((a, b) => a - b);

  // 获取当前的可用宽度
  let validSize = wrapperWidth;
  let breakpoint = false;
  for (const size of sizeList) {
    if (wrapperWidth <= size) {
      validSize = size;
      breakpoint = true;
      break;
    }
  }

  // 非断点，返回设置的宽度
  if (!breakpoint) return initWidth;

  // 断点模式，计算当前断点下的宽度
  const col = breakpoints[validSize]!.rowPerView;
  if (hasAroundGutter) return (wrapperWidth - gutter) / col - gutter;
  else return (wrapperWidth - (col - 1) * gutter) / col;
};

export function useCalculateCols(props: WaterfallProps) {
  const wrapperWidth = ref<number>(0);
  const waterfallWrapper = ref<Nullable<any>>(null);

  // 监听窗口尺寸变化
  useResizeObserver(waterfallWrapper, (entries) => {
    const entry = entries[0];
    const { width } = entry.contentRect;
    wrapperWidth.value = width;
  });

  // 列实际宽度
  const colWidth = computed(() => {
    return getItemWidth({
      wrapperWidth: wrapperWidth.value,
      breakpoints: props.breakpoints,
      gutter: props.gutter,
      hasAroundGutter: props.hasAroundGutter,
      initWidth: props.width,
    });
  });

  // 列
  const cols = computed(() => {
    const offset = props.hasAroundGutter ? -props.gutter : props.gutter;
    return Math.floor(
      (wrapperWidth.value + offset) / (colWidth.value + props.gutter)
    );
  });

  // 偏移
  const offsetX = computed(() => {
    // 左对齐
    if (props.align === "left") {
      return 0;
    } else if (props.align === "center") {
      // 居中
      const offset = props.hasAroundGutter ? props.gutter : -props.gutter;
      const contextWidth =
        cols.value * (colWidth.value + props.gutter) + offset;
      return (wrapperWidth.value - contextWidth) / 2;
    } else {
      const offset = props.hasAroundGutter ? props.gutter : -props.gutter;
      const contextWidth =
        cols.value * (colWidth.value + props.gutter) + offset;
      return wrapperWidth.value - contextWidth;
    }
  });

  return {
    waterfallWrapper,
    wrapperWidth,
    colWidth,
    cols,
    offsetX,
  };
}

const transform = prefixStyle("transform");
const duration = prefixStyle("animation-duration");
const delay = prefixStyle("animation-delay");
const transition = prefixStyle("transition");
const fillMode = prefixStyle("animation-fill-mode");

export function useLayout(
  props: WaterfallProps,
  colWidth: Ref<number>,
  cols: Ref<number>,
  offsetX: Ref<number>,
  waterfallWrapper: Ref<Nullable<HTMLElement>>
) {
  const posY = ref<number[]>([]);
  const wrapperHeight = ref(0);
  // 单个图片高度
  const itemHeight = ref(0);

  // 获取对应y下标的x的值
  const getX = (index: number): number => {
    const count = props.hasAroundGutter ? index + 1 : index;
    return props.gutter * count + colWidth.value * index + offsetX.value;
  };

  // 初始y
  const initY = (): void => {
    posY.value = new Array(cols.value).fill(
      props.hasAroundGutter ? props.gutter : 0
    );
  };

  // 添加入场动画
  const animation = addAnimation(props);

  // 排版
  const layoutHandle = async (): Promise<boolean> => {
    return new Promise((resolve) => {
      // 初始化y集合
      initY();

      // 构造列表
      const items: HTMLElement[] = [];
      if (waterfallWrapper && waterfallWrapper.value) {
        waterfallWrapper.value.childNodes.forEach((el: any) => {
          if (el!.className === "waterfall-item") items.push(el);
        });
      }
      // 获取节点
      if (items.length === 0) return false;

      // 遍历节点
      for (const element of items) {
        const curItem = element as HTMLElement;
        // 最小的y值
        const minY = Math.min.apply(null, posY.value);
        // 最小y的下标
        const minYIndex = posY.value.indexOf(minY);
        // 当前下标对应的x
        const curX = getX(minYIndex);

        // 设置x,y,width
        const style = curItem.style as CssStyleObject;

        // 设置偏移
        if (transform) style[transform] = `translate(${curX}px,${minY}px)`;
        style.width = `${colWidth.value}px`;

        style.visibility = "visible";

        // 更新当前index的y值 获取当前元素的高度
        const { height } = curItem.getBoundingClientRect();
        itemHeight.value = height;

        posY.value[minYIndex] += height + props.gutter;

        // 添加入场动画
        animation(curItem, () => {
          // 添加动画时间
          const time = props.posDuration / 1000;
          if (transition) style[transition] = `transform ${time}s`;
        });
      }

      wrapperHeight.value = Math.max.apply(null, posY.value);

      setTimeout(() => {
        resolve(true);
      }, props.posDuration);
    });
  };

  return {
    wrapperHeight,
    itemHeight,
    layoutHandle,
  };
}

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
