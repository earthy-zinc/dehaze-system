import React from "react";
import { View, Text, Image } from "@tarojs/components";

import "./ComparisonItem.less";

interface ComparisonItemProps {
  imageUrl: string;
}

const ComparisonItem: React.FC<ComparisonItemProps> = ({ imageUrl }) => {
  return (
    <View className="comparison-item">
      <Image
        src={imageUrl}
        className="showcase-image"
        mode="widthFix"
        lazyLoad
      />
      <View className="comparison-label">
        <Text className="label-before">去雾前</Text>
        <Text className="label-divider">→</Text>
        <Text className="label-after">去雾后</Text>
      </View>
    </View>
  );
};

export default ComparisonItem;
