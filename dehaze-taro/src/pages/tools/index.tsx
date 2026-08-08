/**
 * 工具 Tab 根页面（重构）
 *
 * 按 05 规划 2.2：功能聚合中心 —— 页内搜索 + 快捷入口横滑 + 功能网格
 */
import React, { useCallback } from "react";
import { View, Text, ScrollView } from "@tarojs/components";
import Taro from "@tarojs/taro";
import {
  Clock,
  PhotoOutlined,
  BarChartOutlined,
  Arrow,
  DescriptionOutlined,
} from "@taroify/icons";
import SearchBar from "@/components/common/SearchBar";
import PageLayout from "@/layout";
import "./index.less";

interface QuickEntry {
  label: string;
  icon: React.ReactNode;
  target: string;
}

interface GridEntry {
  label: string;
  icon: React.ReactNode;
  target: string;
  isTab?: boolean;
}

const quickEntries: QuickEntry[] = [
  {
    label: "处理历史",
    icon: <Clock size="20" color="#3b82f6" />,
    target: "/pages/task/index",
  },
  {
    label: "我的收藏",
    icon: <BarChartOutlined size="20" color="#3b82f6" />,
    target: "/pages/favorite/index",
  },
  {
    label: "批量处理",
    icon: <Arrow size="20" color="#3b82f6" />,
    target: "/pages/task/index",
  },
  {
    label: "算法选择",
    icon: <BarChartOutlined size="20" color="#3b82f6" />,
    target: "/pages/algorithm-select/index",
  },
];

const gridEntries: GridEntry[] = [
  {
    label: "图像输入",
    icon: <PhotoOutlined size="24" color="#3b82f6" />,
    target: "/pages/image-input/index",
  },
  {
    label: "算法库",
    icon: <BarChartOutlined size="24" color="#3b82f6" />,
    target: "/pages/algorithm/index",
  },
  {
    label: "数据集",
    icon: <BarChartOutlined size="24" color="#3b82f6" />,
    target: "/pages/dataset/index",
  },
  {
    label: "批量处理",
    icon: <Arrow size="24" color="#3b82f6" />,
    target: "/pages/task/index",
  },
  {
    label: "指标管理",
    icon: <BarChartOutlined size="24" color="#3b82f6" />,
    target: "/pages/metrics/index",
  },
  {
    label: "API 文档",
    icon: <DescriptionOutlined size="24" color="#3b82f6" />,
    target: "",
  },
];

const ToolsPage: React.FC = () => {
  const handleSearch = useCallback((value: string) => {
    if (!value) return;
    Taro.showToast({ title: `搜索「${value}」`, icon: "none" });
  }, []);

  const handleGridClick = (entry: GridEntry) => {
    if (entry.label === "API 文档") {
      Taro.showToast({ title: "API 文档功能开发中，敬请期待", icon: "none" });
      return;
    }
    if (entry.isTab) {
      Taro.switchTab({ url: entry.target });
    } else {
      Taro.navigateTo({ url: entry.target });
    }
  };

  const handleQuickClick = (entry: QuickEntry) => {
    Taro.navigateTo({ url: entry.target });
  };

  return (
    <PageLayout level="L1" title="工具">
      <View className="tools-page">
        {/* 全局搜索栏 */}
        <View className="tools-search">
          <SearchBar
            placeholder="搜索算法、功能、文档..."
            onSearch={handleSearch}
          />
        </View>

        {/* 快捷入口横滑区 */}
        <View className="tools-quick-section">
          <Text className="tools-section-label">快捷入口</Text>
          <ScrollView scrollX className="tools-quick-scroll">
            <View className="tools-quick-row">
              {quickEntries.map((entry) => (
                <View
                  key={entry.label}
                  className="tools-quick-item"
                  onClick={() => handleQuickClick(entry)}
                >
                  <View className="tools-quick-icon">{entry.icon}</View>
                  <Text className="tools-quick-label">{entry.label}</Text>
                </View>
              ))}
            </View>
          </ScrollView>
        </View>

        {/* 功能网格 */}
        <View className="tools-grid-section">
          <Text className="tools-section-label">全部功能</Text>
          <View className="tools-grid">
            {gridEntries.map((entry) => (
              <View
                key={entry.label}
                className="tools-grid-item"
                onClick={() => handleGridClick(entry)}
              >
                <View className="tools-grid-icon">{entry.icon}</View>
                <Text className="tools-grid-label">{entry.label}</Text>
              </View>
            ))}
          </View>
        </View>
      </View>
    </PageLayout>
  );
};

export default ToolsPage;
