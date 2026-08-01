import { HeartFilled, HeartOutlined, LoadingOutlined } from "@ant-design/icons";
import { Button, Tooltip } from "antd";
import React from "react";
import { useFavorite } from "@/hooks/useFavorite";
import "./index.module.scss";

export type FavoriteSize = "small" | "middle" | "large";

interface FavoriteButtonProps {
  targetType: "algorithm" | "result" | "dataset" | "image" | "preset";
  targetId: number;
  size?: FavoriteSize;
  showText?: boolean;
  className?: string;
}

const SIZE_MAP: Record<
  FavoriteSize,
  { width: number; height: number; fontSize: number }
> = {
  small: { width: 28, height: 28, fontSize: 14 },
  middle: { width: 36, height: 36, fontSize: 18 },
  large: { width: 44, height: 44, fontSize: 22 },
};

const TEXT_MAP = {
  algorithm: "收藏算法",
  result: "收藏结果",
  dataset: "收藏数据集",
  image: "收藏图片",
  preset: "收藏预设",
};

const FavoriteButton: React.FC<FavoriteButtonProps> = ({
  targetType,
  targetId,
  size = "middle",
  showText = false,
  className = "",
}) => {
  const { isFavorited, loading, toggle } = useFavorite(targetType, targetId);
  const s = SIZE_MAP[size];

  return (
    <Tooltip
      title={isFavorited ? "已收藏" : TEXT_MAP[targetType]}
      placement="top"
    >
      <Button
        type={isFavorited ? "primary" : "text"}
        className={`favorite-button ${className}`}
        style={{
          width: s.width,
          height: s.height,
          padding: 0,
          borderRadius: "50%",
        }}
        onClick={toggle}
        loading={loading}
        icon={
          loading ? (
            <LoadingOutlined spin />
          ) : isFavorited ? (
            <HeartFilled style={{ fontSize: s.fontSize, color: "#fff" }} />
          ) : (
            <HeartOutlined style={{ fontSize: s.fontSize }} />
          )
        }
      >
        {showText && <span>{isFavorited ? "已收藏" : "收藏"}</span>}
      </Button>
    </Tooltip>
  );
};

export default FavoriteButton;
