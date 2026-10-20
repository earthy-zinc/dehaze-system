/**
 * 通用状态标签组件
 * status=1 → 启用（success），其他 → 禁用（danger）
 */
import React from "react";
import { Tag } from "@taroify/core";

interface StatusTagProps {
  status?: number;
  size?: "small" | "medium" | "large";
}

const StatusTag: React.FC<StatusTagProps> = ({ status, size = "small" }) => {
  return status === 1 ? (
    <Tag color="success" size={size}>
      启用
    </Tag>
  ) : (
    <Tag color="danger" size={size}>
      禁用
    </Tag>
  );
};

export default StatusTag;
