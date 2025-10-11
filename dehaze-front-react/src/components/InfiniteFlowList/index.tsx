import { useWindowSize } from "@/hooks/useWindowSize";
import { Timeout } from "ahooks/lib/useRequest/src/types";
import React, { useCallback, useEffect, useRef, useState } from "react";
import PropTypes from "prop-types";

interface InfiniteFlowListProps {
  urls: string[];
  type?: "down" | "up";
  scrollSpeed?: number;
}

// 计算初始滚动位置的辅助函数
const calculateOriginStatus = (
  scrollDirection: "down" | "up",
  itemLength: number,
  itemHeight: number
) => {
  return scrollDirection === "down"
    ? -2 * itemLength * itemHeight + window.innerHeight
    : 0;
};

// 获取滚动方向调整值的辅助函数
const getAdjustment = (scrollDirection: "down" | "up") =>
  scrollDirection === "down" ? 1 : -1;

// 判断是否需要重置滚动索引的辅助函数
const shouldResetIndex = (
  nextIndex: number,
  scrollDirection: "down" | "up",
  itemLength: number,
  itemHeight: number
) => {
  return (
    (scrollDirection === "down" &&
      nextIndex >= -itemLength * itemHeight + window.innerHeight) ||
    (scrollDirection !== "down" && nextIndex <= -itemLength * itemHeight)
  );
};

// 计算下一个滚动索引的辅助函数
const getNextIndex = (
  prevIndex: number,
  adjustment: number,
  scrollDirection: "down" | "up",
  itemLength: number,
  itemHeight: number
) => {
  const nextIndex = prevIndex + adjustment;
  return shouldResetIndex(nextIndex, scrollDirection, itemLength, itemHeight)
    ? calculateOriginStatus(scrollDirection, itemLength, itemHeight)
    : nextIndex;
};

const InfiniteFlowList: React.FC<InfiniteFlowListProps> = ({
  urls,
  type = "down",
  scrollSpeed = 15,
}) => {
  const { width, height } = useWindowSize();
  const [scrollIndex, setScrollIndex] = useState(0);
  const itemHeight = height * 0.4;
  const doubleItems = [...urls, ...urls];
  const timerRef = useRef<Timeout | null>(null);

  // 抽取滚动逻辑到单独的函数
  const startScrolling = useCallback(() => {
    const originStatus = calculateOriginStatus(
      type,
      itemHeight,
      window.innerHeight
    );
    setScrollIndex(originStatus);
    timerRef.current = setInterval(() => {
      const adjustment = getAdjustment(type);
      setScrollIndex((prev) =>
        getNextIndex(prev, adjustment, type, urls.length, itemHeight)
      );
    }, scrollSpeed);
  }, [type, itemHeight, scrollSpeed, urls.length]);

  useEffect(() => {
    startScrolling();
    return () => clearInterval(timerRef.current!);
  }, [urls, itemHeight, type, scrollSpeed, startScrolling]);

  function handleMouseOut() {
    if (!timerRef.current) {
      timerRef.current = setInterval(() => {
        const adjustment = type === "down" ? 1 : -1;
        setScrollIndex((prev) => {
          const nextIndex = prev + adjustment;
          if (
            (type === "down" &&
              nextIndex >= -urls.length * itemHeight + window.innerHeight) ||
            (type !== "down" && nextIndex <= -urls.length * itemHeight)
          ) {
            return -urls.length * itemHeight * (type === "down" ? 2 : 1);
          }
          return nextIndex;
        });
      }, scrollSpeed);
    }
  }

  function handleMouseOver() {
    clearInterval(timerRef.current!);
    timerRef.current = null;
  }

  return (
    <div
      className="scroll-container"
      onMouseOut={handleMouseOut}
      onMouseOver={handleMouseOver}
    >
      <div
        style={{
          transform: `translateY(${scrollIndex}px)`,
          height: `${doubleItems.length * itemHeight}px`,
        }}
        className="scroll-content"
      >
        {doubleItems.map((url, index) => (
          <div key={index} className="scroll-item">
            <img src={url} alt="数据集图片" />
          </div>
        ))}
      </div>
    </div>
  );
};

InfiniteFlowList.propTypes = {
  urls: PropTypes.arrayOf(PropTypes.string.isRequired).isRequired,
  type: PropTypes.oneOf(["down", "up"]),
  scrollSpeed: PropTypes.number,
};

export default InfiniteFlowList;
