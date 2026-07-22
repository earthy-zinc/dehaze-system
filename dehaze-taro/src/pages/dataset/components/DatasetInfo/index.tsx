import React from "react";
import { View, Text } from "@tarojs/components";
import type { Dataset } from "../../services/types";
import "./DatasetInfo.less";

interface DatasetInfoProps {
  dataset: Dataset;
  className?: string;
}

const DatasetInfo: React.FC<DatasetInfoProps> = ({
  dataset,
  className = "",
}) => {
  const stats = dataset.statistics;
  const fileCount = stats?.fileCount || dataset.total || 0;
  const annotatedCount = stats?.annotatedCount || 0;
  const unannotatedCount = stats?.unannotatedCount || 0;

  const formatDate = (date?: string | Date) => {
    if (!date) return "-";
    return new Date(date).toLocaleDateString("zh-CN");
  };

  return (
    <View className={`dataset-info ${className}`}>
      <View className="info-header">
        <Text className="dataset-title">{dataset.name}</Text>
        <Text className="dataset-subtitle">
          {dataset.description || "暂无描述"}
        </Text>
      </View>

      <View className="stats-grid">
        <View className="stat-box">
          <Text className="stat-value">{fileCount.toLocaleString()}</Text>
          <Text className="stat-label">总计图片</Text>
        </View>
        <View className="stat-box">
          <Text className="stat-value">{annotatedCount.toLocaleString()}</Text>
          <Text className="stat-label">已标注</Text>
        </View>
        <View className="stat-box">
          <Text className="stat-value">
            {unannotatedCount.toLocaleString()}
          </Text>
          <Text className="stat-label">未标注</Text>
        </View>
      </View>

      <View className="meta-info">
        <View className="meta-item">
          <Text className="meta-label">创建时间:</Text>
          <Text className="meta-value">{formatDate(dataset.createTime)}</Text>
        </View>
        <View className="meta-item">
          <Text className="meta-label">更新时间:</Text>
          <Text className="meta-value">{formatDate(dataset.updateTime)}</Text>
        </View>
      </View>
    </View>
  );
};

export default DatasetInfo;
