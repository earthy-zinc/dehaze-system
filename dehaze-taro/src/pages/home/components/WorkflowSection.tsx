import React from "react";
import { View, Text } from "@tarojs/components";
import Taro from "@tarojs/taro";
import { PhotoOutlined, BulbOutlined, Fire, Arrow } from "@taroify/icons";

import "./WorkflowSection.less";

interface WorkflowStepProps {
  number: string;
  icon: React.ReactNode;
  title: string;
  description: string;
  target: string;
}

const WorkflowStep: React.FC<WorkflowStepProps> = ({
  number,
  icon,
  title,
  description,
  target,
}) => {
  const handleClick = () => {
    // algorithm-select 和 processing 依赖 current_image，未选择图片时引导用户先去图像输入
    if (target === "algorithm-select" || target === "processing") {
      let hasImage = false;
      try {
        hasImage = !!Taro.getStorageSync("current_image");
      } catch {
        hasImage = false;
      }
      if (!hasImage) {
        Taro.showToast({ title: "请先选择图片", icon: "none" });
        setTimeout(() => Taro.navigateTo({ url: "/pages/image-input/index" }), 1000);
        return;
      }
    }
    Taro.navigateTo({ url: `/pages/${target}/index` });
  };

  return (
    <View className="workflow-step" onClick={handleClick}>
      <View className="step-number">{number}</View>
      <View className="step-icon">{icon}</View>
      <Text className="step-title">{title}</Text>
      <View className="step-desc">
        {description.split("\n").map((line) => (
          <Text key={line}>{line}</Text>
        ))}
      </View>
    </View>
  );
};

const WorkflowSection: React.FC = () => {
  return (
    <View className="workflow-section">
      <View className="features-header">
        <Text className="section-title">强大的功能生态</Text>
        <Text className="section-subtitle">从输入到输出，每一步都精心设计</Text>
      </View>

      <View className="workflow-container">
        <WorkflowStep
          number="01"
          icon={<PhotoOutlined size="28" color="#ffffff" />}
          title="图像输入"
          description="支持上传、拍照、样例图片\n多种输入方式随心选择"
          target="image-input"
        />

        <View className="workflow-arrow">
          <Arrow size="20" color="#d1d5db" />
        </View>

        <WorkflowStep
          number="02"
          icon={<BulbOutlined size="28" color="#ffffff" />}
          title="智能算法"
          description="多种去雾算法可选\nAI智能推荐最优方案"
          target="algorithm-select"
        />

        <View className="workflow-arrow">
          <Arrow size="20" color="#d1d5db" />
        </View>

        <WorkflowStep
          number="03"
          icon={<Fire size="28" color="#ffffff" />}
          title="一键处理"
          description="毫秒级处理速度\n实时预览处理效果"
          target="processing"
        />
      </View>
    </View>
  );
};

export default WorkflowSection;
