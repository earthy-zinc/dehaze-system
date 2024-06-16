import {
  Breakpoints,
  ViewCard,
  WaterfallProps,
} from "@/components/Waterfall/types.ts";
import "./index.scss";
import { Nullable } from "@/utils/types.ts";
import React, { useEffect, useState } from "react";
import { getValue } from "@/utils";
import useLayout from "./useLayout";
import useCalculateCols from "./useCalculateCols";
import { useDebounceFn } from "ahooks";
import LazyImg from "../LazyImg";

interface ExtendedWaterfallProps extends WaterfallProps {
  rowKey?: string;
  imgSelector?: string;
  delay?: number;
  crossOrigin?: boolean;
  lazyload?: boolean;
  loadProps: Object;
  children: React.ReactNode;
}

const Waterfall: React.FC<ExtendedWaterfallProps> = ({
  list = [],
  rowKey = "id",
  imgSelector = "src",
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
  lazyload = true,
  loadProps = {},
  crossOrigin = true,
  delay = 300,
  align = "center",
}) => {
  const waterfallWrapper = React.useRef<Nullable<HTMLDivElement>>(null);

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

  const renderer = useDebounceFn(
    () => {
      layoutHandle();
    },
    {
      wait: delay,
    }
  );

  useEffect(() => {
    if (wrapperWidth > 0) renderer.run();
  }, [wrapperWidth, colWidth, list, renderer]);

  const getRenderURL = (item: ViewCard): string => {
    return getValue(item, imgSelector ?? "src")[0];
  };

  const getKey = (item: ViewCard, index: number): string => {
    return item[rowKey ?? "id"] || index.toString();
  };

  return (
    <div
      ref={waterfallWrapper}
      className="waterfall-list"
      style={{ height: `${wrapperHeight}px` }}
    >
      {list.map((item, index) => (
        <div key={getKey(item, index)} className="waterfall-item">
          <div className="waterfall-card">
            <LazyImg />
          </div>
        </div>
      ))}
    </div>
  );
};

export default Waterfall;
