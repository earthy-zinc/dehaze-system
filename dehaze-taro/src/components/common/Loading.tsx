import React from "react";
import { View } from "@tarojs/components";
import { Loading as TaroLoading } from "@taroify/core";
import "./Loading.less";

interface LoadingProps {
  children?: React.ReactNode;
  size?: "small" | "medium" | "large";
  color?: string;
  className?: string;
  vertical?: boolean;
}

const Loading: React.FC<LoadingProps> = ({
  children = "加载中...",
  size = "medium",
  className = "",
  vertical = true,
}) => {
  return (
    <View className={`loading-container ${className}`}>
      <TaroLoading size={size} direction={vertical ? "vertical" : "horizontal"}>
        {children}
      </TaroLoading>
    </View>
  );
};

export default Loading;
