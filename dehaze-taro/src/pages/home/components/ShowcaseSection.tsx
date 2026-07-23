import React, { useState, useEffect } from "react";
import { View, Text } from "@tarojs/components";
import { DatasetItemAPI } from "dehaze-sdk-js";
import type { DatasetItemVO } from "dehaze-sdk-js";
import ComparisonItem from "./ComparisonItem";
import "./ShowcaseSection.less";

const ShowcaseSection: React.FC = () => {
  const [showcaseImageUrl, setShowcaseImageUrl] = useState("");

  useEffect(() => {
    DatasetItemAPI.getList({
      pageNum: 1,
      pageSize: 1,
      sortBy: "usageCount",
      sortOrder: "desc",
    })
      .then((res) => {
        const item = (res.list as unknown as DatasetItemVO[])?.[0];
        const url = item?.hazyImages?.[0]?.url;
        if (url) setShowcaseImageUrl(url);
      })
      .catch(() => {
        /* 样张加载失败不影响页面其他功能 */
      });
  }, []);

  if (!showcaseImageUrl) return null;

  return (
    <View className="showcase-section">
      <View className="showcase-header">
        <Text className="section-title">一键去雾，效果显著</Text>
        <Text className="section-subtitle">
          智能算法自动识别雾霾程度，精准还原图像细节
        </Text>
      </View>
      <View className="comparison-showcase">
        <ComparisonItem imageUrl={showcaseImageUrl} />
      </View>
    </View>
  );
};

export default ShowcaseSection;
