/**
 * 消息 Tab 根页面（占位）
 *
 * 规划（05）：通知中心 —— 消息列表（系统/处理完成/活动）+ 设置入口
 */
import React from "react";
import { View, Text } from "@tarojs/components";
import { Bell } from "@taroify/icons";
import PageLayout from "@/layout";
import "./index.less";

const tabs = ["全部", "系统通知", "处理完成", "活动"];

const MessagesPage: React.FC = () => (
  <PageLayout level="L1" title="消息">
    <View className="messages-page">
      {/* 分类筛选（规划） */}
      <View className="tabs">
        {tabs.map((tab) => (
          <View key={tab} className={`tab ${tab === "全部" ? "active" : ""}`}>
            {tab}
          </View>
        ))}
      </View>

      {/* 消息列表空态 */}
      <View className="empty-state">
        <Bell size="52" color="#d1d5db" />
        <Text className="empty-text">暂无消息</Text>
        <Text className="empty-sub">处理完成、系统通知等将在这里展示</Text>
      </View>

      <View className="dev-tip">
        <Text>消息通知模块建设中，将对接消息通知 API</Text>
      </View>
    </View>
  </PageLayout>
);

export default MessagesPage;
