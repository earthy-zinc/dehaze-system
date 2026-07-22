import React from "react";
import { View, Text } from "@tarojs/components";
import { SwipeCell, Button } from "@taroify/core";
import {
  Arrow,
  PhotoOutlined,
  CalendarOutlined,
  Edit,
  Delete,
  Add,
} from "@taroify/icons";
import type { Dataset } from "../../services/types";
import "./DatasetCard.less";

// 数据集类型标签配置
const TYPE_LABELS: Record<string, { label: string; className: string }> = {
  training: { label: "训练集", className: "type-training" },
  test: { label: "测试集", className: "type-test" },
  user: { label: "用户集", className: "type-user" },
  result: { label: "结果集", className: "type-result" },
};

interface DatasetCardProps {
  dataset: Dataset;
  depth?: number;
  expanded?: boolean;
  hasChildren?: boolean;
  loading?: boolean;
  onClick?: () => void;
  onToggleExpand?: () => void;
  onAddChild?: () => void;
  onEdit?: () => void;
  onDelete?: () => void;
  className?: string;
}

const DatasetCard: React.FC<DatasetCardProps> = ({
  dataset,
  depth = 0,
  expanded = false,
  hasChildren = false,
  loading = false,
  onClick,
  onToggleExpand,
  onAddChild,
  onEdit,
  onDelete,
  className = "",
}) => {
  const formatDate = (dateString?: string | Date) => {
    if (!dateString) return "-";
    const date = new Date(dateString);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));

    if (days === 0) return "今天";
    if (days === 1) return "昨天";
    if (days < 7) return `${days}天前`;

    return date.toLocaleDateString("zh-CN", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
    });
  };

  const fileCount = dataset.statistics?.fileCount || dataset.total || 0;
  const typeConfig = TYPE_LABELS[dataset.type] || {
    label: dataset.type,
    className: "type-default",
  };
  const isDisabled = dataset.status === 0;

  return (
    <SwipeCell>
      <View
        className={`dataset-card ${className} ${isDisabled ? "disabled" : ""}`}
        style={{ paddingLeft: `${12 + depth * 20}px` }}
        onClick={onClick}
      >
        <View className="card-content">
          {/* 展开/收起按钮 */}
          {hasChildren && (
            <View
              className="expand-btn"
              onClick={(e) => {
                e.stopPropagation();
                onToggleExpand?.();
              }}
            >
              {loading ? (
                <View className="expand-loading" />
              ) : (
                <Arrow
                  size="14"
                  color="#9ca3af"
                  style={{
                    transform: expanded ? "rotate(90deg)" : "rotate(0)",
                    transition: "transform 0.2s",
                  }}
                />
              )}
            </View>
          )}
          {!hasChildren && <View className="expand-placeholder" />}

          {/* 图标占位（path 是存储路径非图片URL，不作为缩略图） */}
          <View className="card-icon">
            <PhotoOutlined size="24" color="#9ca3af" />
          </View>

          <View className="card-info">
            <View className="card-header">
              <Text className="dataset-name">{dataset.name}</Text>
              <View className={`type-tag ${typeConfig.className}`}>
                <Text>{typeConfig.label}</Text>
              </View>
              {isDisabled && (
                <View className="status-tag disabled-tag">
                  <Text>禁用</Text>
                </View>
              )}
            </View>
            <Text className="dataset-description">
              {dataset.description || "暂无描述"}
            </Text>
            <View className="dataset-stats">
              <View className="stat-item">
                <PhotoOutlined size="14" color="#9ca3af" />
                <Text className="stat-value">{fileCount}</Text>
              </View>
              <View className="stat-item">
                <CalendarOutlined size="14" color="#9ca3af" />
                <Text className="stat-value">
                  {formatDate(dataset.createTime)}
                </Text>
              </View>
            </View>
          </View>
        </View>
      </View>
      <SwipeCell.Actions side="right">
        {onAddChild && (
          <Button
            variant="contained"
            color="primary"
            onClick={(e) => {
              e.stopPropagation();
              onAddChild();
            }}
          >
            <Add size="14" />
          </Button>
        )}
        {onEdit && (
          <Button
            variant="contained"
            color="warning"
            onClick={(e) => {
              e.stopPropagation();
              onEdit();
            }}
          >
            <Edit size="14" />
          </Button>
        )}
        {onDelete && (
          <Button
            variant="contained"
            color="danger"
            onClick={(e) => {
              e.stopPropagation();
              onDelete();
            }}
          >
            <Delete size="14" />
          </Button>
        )}
      </SwipeCell.Actions>
    </SwipeCell>
  );
};

export default DatasetCard;
