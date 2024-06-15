import {
  ItemWidthProps,
  WaterfallProps,
} from "@/components/Waterfall/types.ts";
import { useResizeObserver } from "@/hooks/useResizeObserver.ts";
import React, { useState } from "react";

const getItemWidth = ({
  breakpoints,
  wrapperWidth,
  gutter,
  hasAroundGutter,
  initWidth,
}: ItemWidthProps) => {
  const sizeList = Object.keys(breakpoints)
    .map(Number)
    .sort((a, b) => a - b);

  let validSize = wrapperWidth;
  let breakpoint = false;
  for (const size of sizeList) {
    if (wrapperWidth <= size) {
      validSize = size;
      breakpoint = true;
      break;
    }
  }

  if (!breakpoint) return initWidth;

  const col = breakpoints[validSize]?.rowPerView;
  return hasAroundGutter
    ? (wrapperWidth - gutter) / col - gutter
    : (wrapperWidth - (col - 1) * gutter) / col;
};

const useCalculateCols = (props: WaterfallProps) => {
  const [wrapperWidth, setWrapperWidth] = useState(0);
  const waterfallWrapper = React.useRef(null);

  const handleResize = (entries: ResizeObserverEntry[]) => {
    const { width } = entries[0].contentRect;
    setWrapperWidth(width);
  };

  useResizeObserver(waterfallWrapper, handleResize);

  const colWidth = React.useMemo(() => {
    return getItemWidth({
      breakpoints: props.breakpoints,
      wrapperWidth,
      gutter: props.gutter,
      hasAroundGutter: props.hasAroundGutter,
      initWidth: props.width,
    });
  }, [
    wrapperWidth,
    props.breakpoints,
    props.gutter,
    props.hasAroundGutter,
    props.width,
  ]);

  const cols = React.useMemo(() => {
    const offset = props.hasAroundGutter ? -props.gutter : props.gutter;
    return Math.floor((wrapperWidth + offset) / (colWidth + props.gutter));
  }, [wrapperWidth, colWidth, props.hasAroundGutter, props.gutter]);

  const offsetX = React.useMemo(() => {
    if (props.align === "left") return 0;
    if (props.align === "center") {
      const offset = props.hasAroundGutter ? props.gutter : -props.gutter;
      const contextWidth = cols * (colWidth + props.gutter) + offset;
      return (wrapperWidth - contextWidth) / 2;
    }
    // align === 'right'
    const offset = props.hasAroundGutter ? props.gutter : -props.gutter;
    const contextWidth = cols * (colWidth + props.gutter) + offset;
    return wrapperWidth - contextWidth;
  }, [
    wrapperWidth,
    colWidth,
    cols,
    props.align,
    props.gutter,
    props.hasAroundGutter,
  ]);

  return {
    waterfallWrapper,
    wrapperWidth,
    colWidth,
    cols,
    offsetX,
  };
};

export default useCalculateCols;
