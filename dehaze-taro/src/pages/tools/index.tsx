/**
 * 工具 Tab 根页面（占位）
 *
 * 规划（05）：功能聚合中心 —— 全局搜索 + 快捷入口横滑 + 功能网格
 */
import React from "react";
import { View, Text, ScrollView } from "@tarojs/components";
import { Search, Clock, AppsOutlined } from "@taroify/icons";
import PageLayout from "@/layout";
import "./index.less";

const quickEntries = ["处理历史", "我的收藏", "批量处理", "算法选择"];
const gridEntries = [
  "数据集",
  "算法库",
  "指标管理",
  "批量处理",
  "图像输入",
  "API文档",
];

const ToolsPage: React.FC = () => (
  <PageLayout level="L1" title="工具">
    <View className="tools-page">
      {/* 搜索栏（规划：全局搜索算法/功能/文档） */}
      <View className="search-bar">
        <Search size="16" color="#9ca3af" />
        <Text className="search-placeholder">搜索算法、功能...</Text>
      </View>

      {/* 快捷入口横滑区（规划：高频功能横滑直达） */}
      <ScrollView scrollX className="quick-scroll">
        <View className="quick-row">
          {quickEntries.map((item) => (
            <View key={item} className="quick-item">
              <View className="quick-icon">
                <Clock size="18" color="#3b82f6" />
              </View>
              <Text className="quick-label">{item}</Text>
            </View>
          ))}
        </View>
      </ScrollView>

      {/* 功能网格（规划：工具/浏览类功能，管理类归「我的」） */}
      <View className="grid-section">
        <Text className="section-label">全部功能</Text>
        <View className="grid">
          {gridEntries.map((item) => (
            <View key={item} className="grid-item">
              <View className="grid-icon">
                <AppsOutlined size="20" color="#3b82f6" />
              </View>
              <Text className="grid-label">{item}</Text>
            </View>
          ))}
        </View>
      </View>

      <View className="dev-tip">
        <Text>工具聚合中心建设中，功能入口将按规划接入</Text>
      </View>
    </View>
  </PageLayout>
);

export default ToolsPage;
