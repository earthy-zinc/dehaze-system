import { RootState } from "@/store";
import {
  setMagnifierZoomLevel,
  setMaskXY,
  setMouseXY,
} from "@/store/modules/imageShowSlice";
import {
  selectDividerEnabled,
  selectMagnifierZoomLevel,
  selectMaskHeight,
  selectMaskWidth,
} from "@/store/selector/imageShowSelector";
import React, { useEffect, useRef, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import styles from "./index.module.scss";

const OverlapImageShow: React.FC = () => {
  const dispatch = useDispatch();
  const urls = useSelector((state: RootState) => state.imageShow.urls);
  const mask = useSelector((state: RootState) => state.imageShow.mask);
  const maskWidth = useSelector(selectMaskWidth);
  const maskHeight = useSelector(selectMaskHeight);
  const zoomLevel = useSelector(selectMagnifierZoomLevel);
  const isMagnifierEnabled = useSelector(
    (state: RootState) => state.imageShow.magnifier.enabled
  );
  const dividerEnabled = useSelector(selectDividerEnabled);
  const themeColor = useSelector(
    (state: RootState) => state.settings.themeColor
  );
  const brightness = useSelector(
    (state: RootState) => state.imageShow.brightness
  );
  const contrast = useSelector((state: RootState) => state.imageShow.contrast);
  const saturate = useSelector((state: RootState) => state.imageShow.saturate);

  const containerRef = useRef<HTMLDivElement>(null);
  const particleCanvasRef = useRef<HTMLCanvasElement>(null);
  const [dividerPosition, setDividerPosition] = useState(50);

  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        dispatch(setMouseXY({ x: rect.x, y: rect.y }));
      }
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const handleMouseMove = (event: React.MouseEvent | React.TouchEvent) => {
    let clientX, clientY;
    if (event instanceof MouseEvent) {
      clientX = event.clientX;
      clientY = event.clientY;
    } else if (event instanceof TouchEvent) {
      clientX = event.touches[0].clientX;
      clientY = event.touches[0].clientY;
    } else {
      return;
    }

    const rect = containerRef.current!.getBoundingClientRect();
    const maskX = clientX - rect.left - maskWidth / 2;
    const maskY = clientY - rect.top - maskHeight / 2;
    dispatch(setMaskXY({ x: maskX, y: maskY }));
    dispatch(setMouseXY({ x: clientX, y: clientY }));
  };

  const handleWheel = (event: React.WheelEvent) => {
    const newZoom = Math.max(1, zoomLevel + (event.deltaY < 0 ? 0.2 : -0.2));
    dispatch(setMagnifierZoomLevel(newZoom));
  };

  const handleDividerDrag = (offsetX: number) => {
    const percentage = (offsetX / containerRef.current!.offsetWidth) * 100;
    setDividerPosition(percentage);
  };

  const renderParticleEffect = () => {
    if (!particleCanvasRef.current) return;
    const ctx = particleCanvasRef.current.getContext("2d")!;
    ctx.clearRect(
      0,
      0,
      particleCanvasRef.current.width,
      particleCanvasRef.current.height
    );
    // 实现粒子动画逻辑（参考 Vue 的 playParticleEffect）
    requestAnimationFrame(renderParticleEffect);
  };

  useEffect(() => {
    if (particleCanvasRef.current) {
      renderParticleEffect();
    }
  }, []);

  return (
    <div
      ref={containerRef}
      className={styles.comparisonContainer}
      onMouseMove={handleMouseMove}
      onTouchMove={handleMouseMove}
      onWheel={handleWheel}
    >
      <canvas ref={particleCanvasRef} className={styles.particleCanvas} />
      {urls.map((urlData, index) => (
        <img
          key={urlData.url}
          src={urlData.url}
          alt={urlData.label.text}
          style={{
            filter: `brightness(${brightness}%) contrast(${contrast}%) saturate(${saturate}%)`,
            clipPath:
              index === 1
                ? `polygon(${dividerPosition}% 0, 100% 100%)`
                : undefined,
            position: "absolute",
            objectFit: "contain",
          }}
        />
      ))}
      {dividerEnabled && (
        <div
          className={styles.divider}
          style={{
            left: `${dividerPosition}%`,
            backgroundColor: themeColor,
          }}
          onTouchStart={(e) => handleDividerDrag(e.touches[0].clientX)}
          onMouseDown={(e) => handleDividerDrag(e.clientX)}
        >
          <div className={styles.label}>
            <span className={styles.leftLabel}>{urls[0].label.text}</span>
            <span className={styles.rightLabel}>{urls[1].label.text}</span>
          </div>
        </div>
      )}
      {isMagnifierEnabled && (
        <div
          className={styles.mask}
          style={{
            left: `${mask.x}px`,
            top: `${mask.y}px`,
            width: `${maskWidth}px`,
            height: `${maskHeight}px`,
            borderRadius: zoomLevel > 3 ? "50%" : "0",
          }}
        />
      )}
    </div>
  );
};

export default OverlapImageShow;
