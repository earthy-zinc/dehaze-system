/**
 * 底部导航栏组件
 */
import React, { useState, useEffect } from "react";
import { View, Text } from "@tarojs/components";
import Taro from "@tarojs/taro";
import {
  HomeOutlined,
  PhotoOutlined,
  BulbOutlined,
  ManagerOutlined,
} from "@taroify/icons";
import { tabBarItems, getTabBarIndex, type MenuItem } from "@/config/menu";
import "./index.less";

interface AppTabbarProps {
  /** 当前选中索引 */
  current?: number;
  /** 当前路由（用于自动计算索引） */
  currentRoute?: string;
  /** 激活颜色 */
  activeColor?: string;
  /** 未激活颜色 */
  inactiveColor?: string;
}

const AppTabbar: React.FC<AppTabbarProps> = ({
  current = 0,
  currentRoute = "/pages/home/index",
  activeColor = "#3b82f6",
  inactiveColor = "#9ca3af",
}) => {
  const [currentIndex, setCurrentIndex] = useState(current);

  // 监听 currentRoute 变化，自动更新索引
  useEffect(() => {
    if (currentRoute) {
      const index = getTabBarIndex(currentRoute);
      setCurrentIndex(index);
    }
  }, [currentRoute]);

  // 监听 current 属性变化
  useEffect(() => {
    setCurrentIndex(current);
  }, [current]);

  /** 切换 Tab */
  const switchTab = (item: MenuItem, index: number) => {
    if (currentIndex === index) return;

    // algorithm-select 依赖 current_image，未选择图片时引导用户先去图像输入
    if (item.route === "/pages/algorithm-select/index") {
      let hasImage = false;
      try {
        hasImage = !!Taro.getStorageSync("current_image");
      } catch {
        hasImage = false;
      }
      if (!hasImage) {
        Taro.showToast({ title: "请先选择图片", icon: "none" });
        setTimeout(() => Taro.reLaunch({ url: "/pages/image-input/index" }), 1000);
        return;
      }
    }

    setCurrentIndex(index);

    // 使用 reLaunch 统一跳转（项目使用自定义 tabbar 组件，未配置原生 tabBar）
    Taro.reLaunch({
      url: item.route,
      fail: () => {
        Taro.showToast({ title: "页面开发中", icon: "none" });
      },
    });
  };

  /** 获取图标组件 */
  const getIconComponent = (icon: string, isActive: boolean) => {
    const color = isActive ? activeColor : inactiveColor;
    const size = "20";

    const iconMap: Record<string, React.ReactNode> = {
      "home-o": <HomeOutlined size={size} color={color} />,
      "wap-home": <HomeOutlined size={size} color={color} />,
      photograph: <PhotoOutlined size={size} color={color} />,
      "bulb-o": <BulbOutlined size={size} color={color} />,
      "manager-o": <ManagerOutlined size={size} color={color} />,
      manager: <ManagerOutlined size={size} color={color} />,
    };
    return iconMap[icon] || <HomeOutlined size={size} color={color} />;
  };

  return (
    <View className="app-tabbar">
      {tabBarItems.map((item, index) => (
        <View
          key={item.route}
          className={`tabbar-item ${currentIndex === index ? "active" : ""}`}
          onClick={() => switchTab(item, index)}
        >
          <View className="tabbar-icon">
            {getIconComponent(item.icon, currentIndex === index)}
          </View>
          <Text
            className="tabbar-label"
            style={{
              color: currentIndex === index ? activeColor : inactiveColor,
            }}
          >
            {item.title}
          </Text>
        </View>
      ))}
    </View>
  );
};

export default AppTabbar;
