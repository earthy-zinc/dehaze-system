import { Point } from "@/components/AlgorithmToolBar/types";
import React, { useRef } from "react";

interface OverlapImageShowProps {
  image1: string;
  image2: string;
  showMask: boolean;
  contrast: number;
  brightness: number;
  onOriginScaleChange: (value: number) => void;
  onMouseover: (point: Point) => void;
}

const OverlapImageShow: React.FC<OverlapImageShowProps> = ({
  image1,
  image2,
  showMask,
  contrast,
  brightness,
  onOriginScaleChange,
  onMouseover,
}) => {
  const maskRef = useRef<HTMLDivElement>(null);
  const mousemove = () => {};
  const mouseleave = () => {};
  const mouseover = () => {};
  return (
    <div
      onMouseLeave={mouseleave}
      onMouseMove={mousemove}
      onMouseOver={mouseover}
      style={{
        position: "relative",
        width: "100%",
        height: "100%",
        overflow: "hidden",
      }}
    >
      <img
        src={image1}
        alt="image1"
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          width: "100%",
          height: "100%",
          objectFit: "cover",
          filter: `brightness(${brightness}%) contrast(${contrast}%)`,
        }}
      />
      <img
        src={image2}
        alt="image2"
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          width: "100%",
          height: "100%",
          objectFit: "cover",
        }}
      />
    </div>
  );
};

export default OverlapImageShow;
