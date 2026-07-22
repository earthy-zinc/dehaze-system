/**
 * 图标兼容性组件
 * 当taroify图标不可用时，使用文本或emoji替代
 */

import React from "react";
import { Text } from "@tarojs/components";

interface IconWrapperProps {
  icon?: string;
  size?: number | string;
  color?: string;
  fallback?: string;
  className?: string;
}

const IconWrapper: React.FC<IconWrapperProps> = ({
  icon,
  size = 24,
  color = "inherit",
  fallback = "📦",
  className = "",
}) => {
  return (
    <Text
      className={className}
      style={{
        fontSize: size,
        color,
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      {icon || fallback}
    </Text>
  );
};

export default IconWrapper;
