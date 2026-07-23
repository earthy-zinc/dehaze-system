import React, { useState, useEffect } from "react";
import { View, Text, Button, Image } from "@tarojs/components";
import Taro from "@tarojs/taro";
import { Arrow, Success } from "@taroify/icons";
import { DatasetItemAPI } from "dehaze-sdk-js";
import type { DatasetItemVO } from "dehaze-sdk-js";

import "./AlgorithmSection.less";

const AlgorithmSection: React.FC = () => {
  const [algorithmImageUrl, setAlgorithmImageUrl] = useState("");

  useEffect(() => {
    DatasetItemAPI.getList({
      pageNum: 1,
      pageSize: 1,
      sortBy: "usageCount",
      sortOrder: "desc",
    })
      .then((res) => {
        const item = (res.list as unknown as DatasetItemVO[])?.[0];
        const url = item?.clearImage?.url;
        if (url) setAlgorithmImageUrl(url);
      })
      .catch(() => {
        /* 样张加载失败不影响页面其他功能 */
      });
  }, []);

  const handleLearnMoreClick = () => {
    Taro.navigateTo({ url: "/pages/algorithm/index" });
  };

  const algorithmFeatures = [
    {
      text: "智能推荐最适合的去雾算法",
    },
    {
      text: "实时对比不同算法的处理效果",
    },
    {
      text: "毫秒级处理速度，即时查看结果",
    },
    {
      text: "支持批量处理和参数自定义",
    },
  ];

  return (
    <View className="algorithm-section">
      <View className="algorithm-content">
        <View className="algorithm-text">
          <Text className="section-title">多算法智能选择</Text>
          <Text className="section-subtitle">
            支持DCP、AOD-Net、DehazeNet等多种先进算法
          </Text>
          <View className="algorithm-features">
            {algorithmFeatures.map((feature) => (
              <View key={feature.text} className="feature-item">
                <Success size="18" color="#34d399" />
                <Text className="feature-text">{feature.text}</Text>
              </View>
            ))}
          </View>
          <Button className="learn-more-btn" onClick={handleLearnMoreClick}>
            了解更多算法详情
            <Arrow className="btn-icon" size="14" color="#3b82f6" />
          </Button>
        </View>
        <View className="algorithm-visual">
          {algorithmImageUrl && (
            <Image
              src={algorithmImageUrl}
              className="algorithm-image"
              mode="widthFix"
              lazyLoad
            />
          )}
        </View>
      </View>
    </View>
  );
};

export default AlgorithmSection;
