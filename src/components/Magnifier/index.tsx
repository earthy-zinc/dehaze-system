import { RootState } from "@/store";
import { LabelType } from "@/store/modules/imageShowSlice";
import {
  selectMaskHeight,
  selectMaskWidth,
  selectScaleX,
  selectScaleY,
} from "@/store/selector/imageShowSelector";
import { loadImage } from "@/utils";
import React, { useCallback, useEffect, useRef, useState } from "react";
import { useSelector } from "react-redux";

interface MagnifierProps {
  src: string;
  label?: LabelType;
}

const drawLabel = (ctx: CanvasRenderingContext2D, label: LabelType) => {
  const { text, color, backgroundColor } = label;
  ctx.font = "15px sans-serif";
  ctx.fillStyle = backgroundColor;
  ctx.globalAlpha = 0.2;

  const metrics = ctx.measureText(text);
  const textWidth = metrics.width + 10;
  ctx.fillRect(0, 0, textWidth, 20);

  ctx.globalAlpha = 1;
  ctx.fillStyle = color;
  ctx.fillText(text, 3, 15);
};

const Magnifier: React.FC<MagnifierProps> = ({ src, label }) => {
  const [img, setImg] = useState<HTMLImageElement>();

  const scaleX = useSelector(selectScaleX);
  const scaleY = useSelector(selectScaleY);

  const mask = useSelector((state: RootState) => state.imageShow.mask);
  const maskWidth = useSelector(selectMaskWidth);
  const maskHeight = useSelector(selectMaskHeight);

  const width = useSelector(
    (state: RootState) => state.imageShow.magnifierInfo.width
  );
  const height = useSelector(
    (state: RootState) => state.imageShow.magnifierInfo.height
  );
  const shape = useSelector(
    (state: RootState) => state.imageShow.magnifierInfo.shape
  );

  const brightness = useSelector(
    (state: RootState) => state.imageShow.imageInfo.brightness
  );
  const contrast = useSelector(
    (state: RootState) => state.imageShow.imageInfo.contrast
  );
  const saturate = useSelector(
    (state: RootState) => state.imageShow.imageInfo.saturate
  );

  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    canvas.width = width;
    canvas.height = height;
    canvas.style.borderRadius = shape === "circle" ? "50%" : "0";
  }, [canvasRef, width, height, shape]);

  useEffect(() => {
    loadImage(src, false).then((image) => setImg(image));
  }, [src]);

  const drawImage = useCallback(() => {
    if (!img || !canvasRef.current) return;
    const ctx = canvasRef.current.getContext("2d")!;
    ctx.filter = `
      brightness(${brightness}%)
      contrast(${contrast}%)
      saturate(${saturate}%)
    `;
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(
      img,
      mask.x * scaleX,
      mask.y * scaleY,
      maskWidth * scaleX,
      maskHeight * scaleY,
      0,
      0,
      width,
      height
    );
    if (label) drawLabel(ctx, label);
  }, [
    img,
    brightness,
    contrast,
    saturate,
    mask,
    maskWidth,
    maskHeight,
    scaleX,
    scaleY,
    width,
    height,
    label,
  ]);

  useEffect(() => {
    drawImage();
  }, [drawImage]);

  return <canvas ref={canvasRef}></canvas>;
};

export default Magnifier;
