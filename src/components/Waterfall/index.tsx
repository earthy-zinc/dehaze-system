import { WaterfallProps } from "@/components/Waterfall/types.ts";
import "./index.scss";
import React, { useEffect, useState } from "react";
import useLayout from "./useLayout";
import useCalculateCols from "./useCalculateCols";
import { useDebounceFn } from "ahooks";
import LazyImg from "../LazyImg";

const Waterfall: React.FC<WaterfallProps> = (props) => {
  const {
    list = [],
    width = 200,
    breakpoints = {
      1200: {
        // when wrapper width < 1200
        rowPerView: 3,
      },
      800: {
        // when wrapper width < 800
        rowPerView: 2,
      },
      500: {
        // when wrapper width < 500
        rowPerView: 1,
      },
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
      {list.map((url) => (
        <div key={url} className="waterfall-item">
          <div className="waterfall-card">
            <LazyImg url={url} renderer={renderer.run} />
          </div>
        </div>
      ))}
    </div>
  );
};

export default Waterfall;
