import { loadImage } from "@/utils";
import { useCallback, useEffect, useRef, useState } from "react";

interface MagnifierProps {
  // 放大镜形状
  shape?: "circle" | "square";
  // 放大镜半径
  radius: number;
  // 放大倍数
  scale: number;
  originScale: number;
  point: {
    x: number;
    y: number;
  };
  src: string;
  bigImgSrc?: string;
}

const Magnifier: React.FC<MagnifierProps> = ({
  shape = "square",
  radius,
  scale,
  originScale,
  point,
  src,
  bigImgSrc,
}) => {
  const [img, setImg] = useState<HTMLImageElement>();
  const [trueScale, setTrueScale] = useState(scale);
  const [swidth, setSwidth] = useState(0);
  const [sheight, setSheight] = useState(0);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const drawImageOnMouseMove = useCallback(() => {
    const ctx = canvasRef.current?.getContext("2d");
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
    }
  }, [point, trueScale, swidth, sheight, img, radius]);

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
    point,
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

  return <canvas ref={canvasRef}></canvas>;
};

export default Magnifier;
