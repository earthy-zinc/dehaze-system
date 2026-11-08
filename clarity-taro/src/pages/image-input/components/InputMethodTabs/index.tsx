/**
 * 输入方式选择组件
 */

import React from "react";
import { Icon, View, Text } from "@/platform/ui";
import { InputMethod } from "../../services/types";
import "./InputMethodTabs.less";

interface InputMethodTabsProps {
  activeMethod: InputMethod;
  onChange: (method: InputMethod) => void;
}

const methodConfig = [
  {
    key: "upload" as InputMethod,
    label: "上传图片",
    subLabel: "从相册选择",
    icon: "photo-o" as const,
  },
  {
    key: "camera" as InputMethod,
    label: "拍照",
    subLabel: "实时拍摄",
    icon: "photograph" as const,
  },
  {
    key: "sample" as InputMethod,
    label: "样例图片",
    subLabel: "快速体验",
    icon: "photo" as const,
  },
  {
    key: "history" as InputMethod,
    label: "历史记录",
    subLabel: "最近处理",
    icon: "manager-o" as const,
  },
];

const InputMethodTabs: React.FC<InputMethodTabsProps> = ({
  activeMethod,
  onChange,
}) => {
  return (
    <View className="input-method-tabs">
      {methodConfig.map((method) => {
        const isActive = activeMethod === method.key;
        return (
          <View
            key={method.key}
            className={`method-tab ${isActive ? "active" : ""}`}
            onClick={() => onChange(method.key)}
          >
            <View className="tab-icon">
              <Icon
                name={method.icon}
                size="28"
                color={isActive ? "#3b82f6" : "#9ca3af"}
              />
            </View>
            <Text className="tab-label">{method.label}</Text>
            <Text className="tab-sub-label">{method.subLabel}</Text>
          </View>
        );
      })}
    </View>
  );
};

export default InputMethodTabs;
