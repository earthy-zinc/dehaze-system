import { loadImage } from "@/utils";
import React, { useCallback, useEffect, useRef, useState } from "react";
import { transform } from "@/components/Magnifier/utils";

interface MagnifierProps {
  // 放大镜形状
  shape?: "circle" | "square";
  // 放大镜半径
  radius: number;
  // 放大倍数
  scale: number;
  originScale: number;
  src: string;
  bigImgSrc?: string;
  brightness?: number;
  contrast?: number;
  point: {
    x: number;
    y: number;
  };
  label?: {
    text: string;
    color: string;
    backgroundColor: string;
  };
}

const Magnifier: React.FC<MagnifierProps> = ({
  shape = "square",
  radius = 100,
  scale = 8,
  originScale,
  point,
  src,
  bigImgSrc,
  brightness = 0,
  contrast = 0,
  label,
}) => {
  const [img, setImg] = useState<HTMLImageElement>();
  const [trueScale, setTrueScale] = useState(scale);
  const [swidth, setSwidth] = useState(0);
  const [sheight, setSheight] = useState(0);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const ctx = canvasRef.current?.getContext("2d");

  const drawLabel = useCallback(() => {
    if (!label || !ctx) return;
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
  }, [label, ctx]);

  const drawImageOnMouseMove = useCallback(() => {
    if (ctx && img) {
      const sx = Math.max(
        0,
        Math.min(point.x / trueScale - sheight / 2, img.width - sheight)
      );
      const sy = Math.max(
        0,
        Math.min(point.y / trueScale - swidth / 2, img.height - swidth)
      );

      ctx.clearRect(0, 0, radius * 2, radius * 2);
      ctx.drawImage(img, sx, sy, swidth, sheight, 0, 0, radius * 2, radius * 2);
      drawLabel();
    }
  }, [
    ctx,
    img,
    point.x,
    point.y,
    trueScale,
    sheight,
    swidth,
    radius,
    drawLabel,
  ]);

  useEffect(() => {
    const init = async () => {
      if (!canvasRef.current) return;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      canvas.width = radius * 2;
      canvas.height = radius * 2;
      canvas.style.borderRadius = shape === "circle" ? "50%" : "0";

      const originImg = await loadImage(src, true);
      setSwidth(originImg.width / scale);
      setSheight(swidth);

      if (bigImgSrc) {
        const bigImg = await loadImage(bigImgSrc, true);
        const enlargeScale = bigImg.width / originImg.width;
        setImg(bigImg);
        setSwidth(swidth * enlargeScale);
        setSheight(sheight * enlargeScale);
        setTrueScale(originScale * enlargeScale);
      } else {
        setTrueScale(originScale);
        setImg(originImg);
      }
      drawImageOnMouseMove();
    };

    init();
  }, [
    src,
    bigImgSrc,
    radius,
    scale,
    shape,
    originScale,
    swidth,
    sheight,
    drawImageOnMouseMove,
  ]);

  useEffect(() => {
    if (!canvasRef.current) return;
    if (!img) return;
    drawImageOnMouseMove();
  }, [point.x, point.y, img, drawImageOnMouseMove]);

  useEffect(() => {
    if (!ctx) return;
    ctx.filter = `brightness(${transform(brightness)}%) contrast(${transform(contrast)}%)`;
  }, [brightness, contrast, ctx]);

  return <canvas ref={canvasRef}></canvas>;
};

export default Magnifier;
