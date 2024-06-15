import {
  Breakpoints,
  ViewCard,
  WaterfallProps,
} from "@/components/Waterfall/types.ts";
import "./index.scss";
import { Nullable } from "@/utils/types.ts";
import React, { useEffect, useState } from "react";
import { getValue } from "@/utils";

interface ExtendedWaterfallProps extends WaterfallProps {
  rowKey?: string;
  imgSelector?: string;
  delay?: number;
  crossOrigin?: boolean;
  lazyload?: boolean;
  loadProps: Object;
  children: React.ReactNode;
}

const Waterfall: React.FC<ExtendedWaterfallProps> = (props) => {
  const {
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
    animationDuration = 1000,
    animationDelay = 300,
    lazyload = true,
    loadProps = {},
    crossOrigin = true,
    delay = 300,
    align = "center",
  } = props;

  const waterfallWrapper = React.useRef<Nullable<HTMLDivElement>>(null);
  const [wrapperHeight, setWrapperHeight] = useState<number>(0);
  const [wrapperWidth, setWrapperWidth] = useState<number>(0);
  const [cols, setCols] = useState<number[]>([]);
  const [colWidth, setColWidth] = useState<number>(0);
  const [offsetX, setOffsetX] = useState<number>(0);

  const getRenderURL = (item: ViewCard): string => {
    return getValue(item, props.imgSelector ?? "src")[0];
  };

  const getKey = (item: ViewCard, index: number): string => {
    return item[props.rowKey ?? "id"] || index.toString();
  };
  return (
    <div
      ref={waterfallWrapper}
      className="waterfall-list"
      style={{ height: `${wrapperHeight}px` }}
    >
      {list.map((item, index) => (
        <div key={getKey(item, index)} className="waterfall-item">
          <div className="waterfall-card"></div>
        </div>
      ))}
    </div>
  );
};

export default Waterfall;
