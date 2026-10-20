import React from "react";
import { View, Text } from "@tarojs/components";
import "./ErrorState.less";

interface ErrorStateProps {
  /** 错误信息 */
  message?: string;
  /** 重试回调，提供则显示重试按钮 */
  onRetry?: () => void;
  /** 重试按钮文案 */
  retryText?: string;
  className?: string;
}

/**
 * 错误状态组件：展示错误信息并提供重试按钮
 */
const ErrorState: React.FC<ErrorStateProps> = ({
  message = "加载失败",
  onRetry,
  retryText = "重试",
  className = "",
}) => {
  return (
    <View className={`error-state ${className}`}>
      <View className="error-icon">
        <Text>⚠️</Text>
      </View>
      <Text className="error-message">{message}</Text>
      {onRetry && (
        <View className="retry-btn" onClick={onRetry}>
          <Text>{retryText}</Text>
        </View>
      )}
    </View>
  );
};

export default ErrorState;
