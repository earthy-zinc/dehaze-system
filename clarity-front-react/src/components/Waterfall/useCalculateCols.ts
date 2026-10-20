import { Breakpoints, ItemWidthProps } from "@/components/Waterfall/types";
import { useResizeObserver } from "@/hooks/useResizeObserver";
import { Nullable } from "@/utils/types";
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

const useCalculateCols = (
  breakpoints: Breakpoints,
  hasAroundGutter: boolean,
  gutter: number,
  width: number,
  align: string,
  waterfallWrapper: React.MutableRefObject<Nullable<HTMLDivElement>>
) => {
  const [wrapperWidth, setWrapperWidth] = useState<number>(0);
  const handleResize = (entries: ResizeObserverEntry[]) => {
    const { width } = entries[0].contentRect;
    setWrapperWidth(width);
  };

  useResizeObserver(waterfallWrapper, handleResize);

  const colWidth = React.useMemo(() => {
    return getItemWidth({
      breakpoints: breakpoints,
      wrapperWidth,
      gutter: gutter,
      hasAroundGutter: hasAroundGutter,
      initWidth: width,
    });
  }, [wrapperWidth, breakpoints, gutter, hasAroundGutter, width]);

  const cols = React.useMemo(() => {
    const offset = hasAroundGutter ? -gutter : gutter;
    return Math.floor((wrapperWidth + offset) / (colWidth + gutter));
  }, [wrapperWidth, colWidth, hasAroundGutter, gutter]);

  const offsetX = React.useMemo(() => {
    const offset = hasAroundGutter ? gutter : -gutter;
    if (align === "left") return 0;
    if (align === "center") {
      const contextWidth = cols * (colWidth + gutter) + offset;
      return (wrapperWidth - contextWidth) / 2;
    }
    // align === 'right'
    const contextWidth = cols * (colWidth + gutter) + offset;
    return wrapperWidth - contextWidth;
  }, [wrapperWidth, colWidth, cols, align, gutter, hasAroundGutter]);

  return {
    wrapperWidth,
    colWidth,
    cols,
    offsetX,
  };
};

export default useCalculateCols;
