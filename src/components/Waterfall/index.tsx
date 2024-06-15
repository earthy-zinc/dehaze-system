import { WaterfallProps } from "@/components/Waterfall/types.ts";
import "./index.scss";
import { Nullable } from "@/utils/types.ts";
import React from "react";

export default function Waterfall(props) {
  const waterfallWrapper = React.useRef<HTMLDivElement>(null);

  return (
    <div ref={waterfallWrapper} className="waterfall-list">
      <img />
    </div>
  );
}
