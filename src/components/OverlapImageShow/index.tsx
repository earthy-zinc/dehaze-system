import { Point } from "@/components/AlgorithmToolBar/types";
import DraggableLine from "@/components/DraggableLine";
import React, { CSSProperties, useEffect, useRef, useState } from "react";

interface OverlapImageShowProps {
  image1: string;
  image2: string;
  showMask?: boolean;
  contrast?: number;
  brightness?: number;
  height?: number;
  onOriginScaleChange: (value: number) => void;
  onMouseover: (point: Point) => void;
}

const OverlapImageShow: React.FC<OverlapImageShowProps> = ({
  image1,
  image2,
  showMask = false,
  contrast = 0,
  brightness = 0,
  height = 700,
  onOriginScaleChange,
  onMouseover,
}) => {
  const image1Ref = useRef<HTMLImageElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [imageSize, setImageSize] = useState<{ width: number; height: number }>(
    { width: 0, height: 0 }
  );
  const maskRef = useRef<HTMLDivElement>(null);
  const [maskStyle, setMaskStyle] = useState<CSSProperties>({
    width: "100px",
    height: "100px",
    backgroundColor: "rgba(0, 0, 0, 0.5)",
    position: "absolute",
    display: "none",
  });
  const [sliderPosition, setSliderPosition] = useState(0);
  const sliderValue =
    (1 - sliderPosition / (image1Ref.current?.width || 1)) * 100;

  useEffect(() => {
    if (image1Ref.current) {
      const naturalHeight = image1Ref.current.naturalHeight;
      const naturalWidth = image1Ref.current.naturalWidth;
      const originScale = naturalHeight / (image1Ref.current.height || 1);

      onOriginScaleChange?.(originScale);
    }
  }, [image1Ref, height, onOriginScaleChange]);

  const mousemove: React.MouseEventHandler<HTMLDivElement> = (event) => {
    const getMaskStyleTopAndLeft = () => {
      let top = event.clientY - (maskRef.current?.offsetHeight || 0) / 2;
      let left = event.clientX - (maskRef.current?.offsetWidth || 0) / 2;

      if (top + (maskRef.current?.offsetHeight || 0) > height) {
        top = height - (maskRef.current?.offsetHeight || 0);
      }
      if (
        left + (maskRef.current?.offsetWidth || 0) >
        (image1Ref.current?.width || 0)
      ) {
        left =
          (image1Ref.current?.width || 0) - (maskRef.current?.offsetWidth || 0);
      }
      if (top < 0) {
        top = 0;
      }
      if (left < 0) {
        left = 0;
      }
      return { maskTop: top, maskLeft: left };
    };

    onMouseover?.({ x: event.clientX, y: event.clientY });

    const { maskTop, maskLeft } = getMaskStyleTopAndLeft();

    setMaskStyle({ ...maskStyle, top: `${maskTop}px`, left: `${maskLeft}px` });
  };

  const mouseover = () => {
    setMaskStyle({ ...maskStyle, display: "block" });
  };

  const mouseleave = () => {
    setMaskStyle({ ...maskStyle, display: "none" });
  };

  return (
    <div
      onMouseLeave={mouseleave}
      onMouseMove={mousemove}
      onMouseOver={mouseover}
      ref={containerRef}
      style={{
        position: "relative",
        width: image1Ref.current?.width || "unset",
        height: image1Ref.current?.height || height,
        overflow: "hidden",
      }}
    >
      <img
        src={image1}
        ref={image1Ref}
        alt="image1"
        style={{
          position: "absolute",
          clipPath: `inset(0 ${sliderValue}% 0 0)`,
          width: "100%",
          height: "100%",
          objectFit: "cover",
          // filter: `brightness(${brightness}%) contrast(${contrast}%)`,
        }}
      />
      <img
        src={image2}
        alt="image2"
        style={{
          position: "absolute",
          clipPath: `inset(0 0 0 ${100 - sliderValue}%)`,
          width: "100%",
          height: "100%",
          objectFit: "cover",
          // filter: `brightness(${brightness}%) contrast(${contrast}%)`,
        }}
      />
      {showMask && (
        <div ref={maskRef} style={maskStyle} className="mouse-mask"></div>
      )}
      <DraggableLine
        leftLabel="原图"
        rightLabel="对比图"
        onUpdateOffset={(value: number) => setSliderPosition(value)}
      />
    </div>
  );
};

export default OverlapImageShow;
