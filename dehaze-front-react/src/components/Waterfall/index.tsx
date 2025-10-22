import { WaterfallProps } from "@/components/Waterfall/types";
import "./index.scss";
import { useDebounceFn } from "ahooks";
import React, { useEffect } from "react";
import LazyImg from "../LazyImg";
import useCalculateCols from "./useCalculateCols";
import useLayout from "./useLayout";

const Waterfall: React.FC<WaterfallProps> = (props) => {
  const {
    list = [],
    width = 300,
    breakpoints = {
      1800: { rowPerView: 4 },
      1200: { rowPerView: 3 },
      800: { rowPerView: 2 },
      500: { rowPerView: 1 },
    },
    gutter = 10,
    hasAroundGutter = true,
    posDuration = 300,
    animationPrefix = "animate__animated",
    animationEffect = "fadeIn",
    animationDelay = 300,
    animationDuration = 1000,
    delay = 300,
    align = "center",
  } = props;
  const waterfallWrapper = React.useRef<HTMLDivElement>(null);

  const { wrapperWidth, colWidth, cols, offsetX } = useCalculateCols(
    breakpoints,
    hasAroundGutter,
    gutter,
    width,
    align,
    waterfallWrapper
  );

  const { wrapperHeight, layoutHandle } = useLayout(
    hasAroundGutter,
    gutter,
    posDuration,
    animationPrefix,
    animationEffect,
    animationDelay,
    animationDuration,
    colWidth,
    cols,
    offsetX,
    waterfallWrapper
  );

  const renderer = useDebounceFn(() => layoutHandle(), {
    wait: delay,
  });

  useEffect(() => {
    if (wrapperWidth > 0) renderer.run();
  }, [wrapperWidth, colWidth, list, renderer]);

  return (
    <div
      ref={waterfallWrapper}
      className="waterfall-list"
      style={{ height: `${wrapperHeight}px` }}
    >
      {list.map((item) => {
        return (
          <div
            key={item.id}
            className="waterfall-item"
            onClick={() => props.onClickItem && props.onClickItem(item.id || 0)}
          >
            <div className="waterfall-card">
              <LazyImg url={item.src} renderer={renderer.run} />
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default Waterfall;
