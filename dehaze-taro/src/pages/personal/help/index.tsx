import React from "react";
import { View, Text, ScrollView } from "@tarojs/components";
import PageLayout from "@/layout";
import "./index.less";

const FAQ_ITEMS = [
  {
    q: "如何使用去雾功能？",
    a: "在「去雾」Tab 页面上传图片，选择算法后即可开始处理。处理完成后可查看对比效果。",
  },
  {
    q: "支持哪些图片格式？",
    a: "支持 JPG、PNG、BMP、TIFF 等常见图像格式，单张图片建议不超过 20MB。",
  },
  {
    q: "如何查看处理历史？",
    a: "在「我的」页面点击「处理历史」，可查看所有处理记录并重新处理。",
  },
  {
    q: "如何获得更多处理次数？",
    a: "可通过升级 VIP 会员获得更多月度处理次数，在「我的会员」页面查看详情。",
  },
  {
    q: "遇到问题如何反馈？",
    a: "在「我的」页面点击「反馈评价」，提交您的问题或建议，我们会尽快回复。",
  },
];

const HelpPage: React.FC = () => {
  return (
    <PageLayout level="L2" title="帮助中心">
      <View className="personal-help-page">
        <ScrollView scrollY className="help-scroll">
          <View className="help-header">
            <Text className="help-header-title">常见问题</Text>
            <Text className="help-header-desc">
              以下是用户经常咨询的问题，如有其他疑问可联系客服
            </Text>
          </View>

          {FAQ_ITEMS.map((item, idx) => (
            <View key={idx} className="help-card">
              <View className="help-card-header">
                <Text className="help-q-icon">Q</Text>
                <Text className="help-question">{item.q}</Text>
              </View>
              <View className="help-answer">
                <Text className="help-a-icon">A</Text>
                <Text className="help-answer-text">{item.a}</Text>
              </View>
            </View>
          ))}
        </ScrollView>
      </View>
    </PageLayout>
  );
};

export default HelpPage;
