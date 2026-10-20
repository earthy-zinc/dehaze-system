import HollowSlide from "@/assets/icons/hollow-slide.svg";
import { useWindowSize } from "@/hooks/useWindowSize";
import { RootState } from "@/store";
import { hexToRGBA } from "@/utils";
import React, { useEffect, useRef, useState } from "react";
import { useSelector } from "react-redux";
import styles from "./index.module.scss";

interface DraggableLineProps {
  leftLabel: string;
  rightLabel: string;
  onUpdateOffset?: (offset: number) => void;
}

const DraggableLine: React.FC<DraggableLineProps> = ({
  leftLabel,
  rightLabel,
  onUpdateOffset,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [parentOffsetLeft, setParentOffsetLeft] = useState(0);
  const [offsetLeft, setOffsetLeft] = useState(0);
  const dragContainerRef = useRef<HTMLDivElement>(null);
  const settingsStore = useSelector((root: RootState) => root.settings);
  const { width } = useWindowSize();

  const handleDragStart = (event: React.MouseEvent<HTMLDivElement>) => {
    setIsDragging(true);
  };

  const handleDrag: React.MouseEventHandler<HTMLDivElement> = (event) => {
    if (isDragging) {
      setOffsetLeft(dragContainerRef.current!.getBoundingClientRect().left);
      setParentOffsetLeft(event.clientX - offsetLeft);
    }
  };

  const handleDragEnd = () => {
    setIsDragging(false);
  };

  useEffect(() => {
    if (onUpdateOffset) {
      onUpdateOffset(parentOffsetLeft);
    }
  }, [parentOffsetLeft, onUpdateOffset]);

  useEffect(() => {
    const animateOffsetLeft = (from: number, to: number, duration: number) => {
      const startTime = performance.now();
      const changeInValue = to - from;

      const animate = (currentTime: number) => {
        const elapsedTime = currentTime - startTime;
        const progress = Math.min(elapsedTime / duration, 1);
        setParentOffsetLeft(from + changeInValue * progress);

        if (progress < 1) {
          requestAnimationFrame(animate);
        }
      };
      requestAnimationFrame(animate);
    };

    if (dragContainerRef.current) {
      const { left, right } = dragContainerRef.current.getBoundingClientRect();
      setParentOffsetLeft(dragContainerRef.current.offsetWidth);
      setOffsetLeft(left);
      animateOffsetLeft(right - left, 0, 3000);
    }
  }, []);

  return (
    <div
      ref={dragContainerRef}
      className={styles.container}
      onMouseMove={handleDrag}
      onMouseUp={handleDragEnd}
      onMouseDown={handleDragStart}
    >
      <div
        style={{
          left: `${parentOffsetLeft}px`,
          backgroundColor: settingsStore.themeColor,
        }}
        className={styles.line}
      >
        <HollowSlide />
      </div>
      {leftLabel && (
        <div
          style={{
            left: `${parentOffsetLeft - 80}px`,
            backgroundColor: "rgba(162,162,162,0.5)",
          }}
          className={styles["drag-label"]}
        >
          <span style={{ color: "#fff" }}> {leftLabel} </span>
        </div>
      )}
      {rightLabel && (
        <div
          style={{
            left: `${parentOffsetLeft}px`,
            backgroundColor: hexToRGBA(settingsStore.themeColor, 0.5),
          }}
          className={styles["drag-label"]}
        >
          <span style={{ color: "#fff" }}> {rightLabel} </span>
        </div>
      )}
    </div>
  );
};

export default DraggableLine;
