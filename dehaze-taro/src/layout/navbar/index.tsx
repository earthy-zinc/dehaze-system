/**
 * 顶部导航栏组件
 */
import React from 'react';
import { View, Text } from '@tarojs/components';
import Taro from '@tarojs/taro';
import { Search, Bars, EyeOutlined } from '@taroify/icons';
import { useStatusBarHeight } from '@/hooks/useLayout';
import './index.less';

interface AppNavbarProps {
  /** 切换菜单回调 */
  onToggleMenu?: () => void;
  /** 搜索回调 */
  onSearch?: () => void;
  /** 标题 */
  title?: string;
}

const AppNavbar: React.FC<AppNavbarProps> = ({
  onToggleMenu,
  onSearch,
  title = '图像去雾系统',
}) => {
  const statusBarHeight = useStatusBarHeight();

  /** 跳转首页 */
  const goHome = () => {
    Taro.switchTab({ url: '/pages/home/index' });
  };

  /** 搜索按钮点击 */
  const handleSearch = () => {
    if (onSearch) {
      onSearch();
    } else {
      Taro.showToast({ title: '搜索功能开发中', icon: 'none' });
    }
  };

  /** 切换菜单 */
  const handleToggleMenu = () => {
    onToggleMenu?.();
  };

  return (
    <View className='app-navbar'>
      {/* 状态栏占位 */}
      <View className='status-bar' style={{ height: `${statusBarHeight}px` }} />

      {/* 导航栏内容 */}
      <View className='navbar-content'>
        {/* Logo + 标题 */}
        <View className='navbar-brand' onClick={goHome}>
          <View className='logo-wrapper'>
            <EyeOutlined size='20' color='#ffffff' />
          </View>
          <Text className='app-title'>{title}</Text>
        </View>

        {/* 右侧操作区 */}
        <View className='navbar-actions'>
          <View className='action-btn' onClick={handleSearch}>
            <Search size='18' color='#374151' />
          </View>
          <View className='action-btn menu-btn' onClick={handleToggleMenu}>
            <Bars size='18' color='#374151' />
          </View>
        </View>
      </View>
    </View>
  );
};

export default AppNavbar;
