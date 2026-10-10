import React, { useState, useEffect } from "react";
import { View, Text, Image } from "@tarojs/components";
import { Success } from "@taroify/icons";
import { DatasetItemAPI } from "dehaze-sdk-js";

import "./ModelSection.less";

const ModelSection: React.FC = () => {
  const [modelImageUrl, setModelImageUrl] = useState("");

  useEffect(() => {
    DatasetItemAPI.getList({
      pageNum: 1,
      pageSize: 1,
      sortBy: "usageCount",
      sortOrder: "desc",
    })
      .then((res) => {
        const item = res.list?.[0];
        const url = item?.clearImage?.url;
        if (url) setModelImageUrl(url);
      })
      .catch(() => {
        /* 样张加载失败不影响页面其他功能 */
      });
  }, []);

  const modelFeatures = [
    {
      text: "智能推荐最适合的去雾模型",
    },
    {
      text: "实时对比不同模型的处理效果",
    },
    {
      text: "毫秒级处理速度，即时查看结果",
    },
    {
      text: "支持批量处理和参数自定义",
    },
  ];

  return (
    <View className="model-section">
      <View className="model-content">
        <View className="model-text">
          <Text className="section-title">多模型智能选择</Text>
          <Text className="section-subtitle">
            平台统一管理的去雾模型，按场景灵活选用
          </Text>
          <View className="model-features">
            {modelFeatures.map((feature) => (
              <View key={feature.text} className="feature-item">
                <Success size="18" color="#34d399" />
                <Text className="feature-text">{feature.text}</Text>
              </View>
            ))}
          </View>
        </View>
        <View className="model-visual">
          {modelImageUrl && (
            <Image
              src={modelImageUrl}
              className="model-image"
              mode="widthFix"
              lazyLoad
            />
          )}
        </View>
      </View>
    </View>
  );
};

export default ModelSection;
