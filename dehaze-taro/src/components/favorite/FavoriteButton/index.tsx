import React from "react";
import { View } from "@tarojs/components";
import { Star, StarOutlined } from "@taroify/icons";
import type { FavoriteTargetType } from "dehaze-sdk-js";
import { useFavorite } from "@/hooks/useFavorite";
import "./index.less";

interface FavoriteButtonProps {
  /** 收藏对象类型 */
  targetType: FavoriteTargetType;
  /** 收藏对象 ID */
  targetId: number;
  /** 图标大小，默认 20 */
  size?: number;
  /** 是否显示文字标签 */
  showLabel?: boolean;
  /** 点击回调 */
  onClick?: (isFavorited: boolean) => void;
}

const FavoriteButton: React.FC<FavoriteButtonProps> = ({
  targetType,
  targetId,
  size = 20,
  showLabel = false,
  onClick,
}) => {
  const { isFavorited, toggle } = useFavorite(targetType, targetId);

  const handleTap = async () => {
    await toggle();
    onClick?.(isFavorited);
  };

  return (
    <View className="favorite-button" onClick={handleTap}>
      {isFavorited ? (
        <Star size={size} color="#f59e0b" />
      ) : (
        <StarOutlined size={size} color="#9ca3af" />
      )}
      {showLabel && (
        <View className="fav-label">{isFavorited ? "已收藏" : "收藏"}</View>
      )}
    </View>
  );
};

export default FavoriteButton;
