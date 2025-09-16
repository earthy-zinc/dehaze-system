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

const drawLabel = (ctx: CanvasRenderingContext2D, label: LabelType): void => {
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
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // 状态获取
  const scaleX = useSelector(selectScaleX);
  const scaleY = useSelector(selectScaleY);
  const mask = useSelector((state: RootState) => state.imageShow.mask);
  const maskWidth = useSelector(selectMaskWidth);
  const maskHeight = useSelector(selectMaskHeight);
  const magnifierInfo = useSelector(
    (state: RootState) => state.imageShow.magnifier
  );
  const brightness = useSelector(
    (state: RootState) => state.imageShow.brightness
  );
  const contrast = useSelector((state: RootState) => state.imageShow.contrast);
  const saturate = useSelector((state: RootState) => state.imageShow.saturate);

  // 从 state 中解构
  const { width, height, shape } = magnifierInfo;

  // 初始化 Canvas 尺寸和样式
  const initCanvas = useCallback(() => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    canvas.width = width;
    canvas.height = height;
    canvas.style.borderRadius = shape === "circle" ? "50%" : "0";
  }, [width, height, shape]);

  // 图像加载
  useEffect(() => {
    if (!src) return;
    loadImage(src, false).then((image) => setImg(image));
  }, [src]);

  // 绘制逻辑
  const drawImage = useCallback(() => {
    if (!canvasRef.current || !img) return;
    const ctx = canvasRef.current.getContext("2d");
    if (!ctx) return;

    // 清除并应用滤镜
    ctx.filter = `brightness(${brightness}%) contrast(${contrast}%) saturate(${saturate}%)`;
    ctx.clearRect(0, 0, width, height);

    // 绘制放大区域
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

    // 绘制标签
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
    shape,
    label,
  ]);

  // 效果初始化
  useEffect(() => {
    initCanvas();
    drawImage();
  }, [initCanvas, drawImage]);

  // 监听尺寸变化
  useEffect(() => {
    initCanvas();
    drawImage();
  }, [width, height, shape, initCanvas, drawImage]);

  // 监听图像变化
  useEffect(() => {
    if (img) drawImage();
  }, [img, drawImage]);

  // 监听状态变化
  useEffect(() => {
    drawImage();
  }, [drawImage]);

  return <canvas ref={canvasRef} />;
};

export default Magnifier;
