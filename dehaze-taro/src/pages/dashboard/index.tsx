import { Text, View } from '@tarojs/components';
import Taro from '@tarojs/taro';
import React from 'react';
import { User, SettingOutlined, ShieldOutlined } from '@taroify/icons';

import './index.less';
import { usePermission } from '@/hooks/usePermission';

const Dashboard: React.FC = () => {

  const { hasPermission } = usePermission();
  // 导航到用户管理
  const navigateToUserManagement = () => {
    Taro.navigateTo({
      url: '/pages/system/user/index'
    });
  };

  // 导航到角色管理
  const navigateToRoleManagement = () => {
    Taro.navigateTo({
      url: '/pages/system/role/index'
    });
  };

  return (
    <View className='dashboard-container'>
      <Text className='welcome-text'>欢迎使用系统管理</Text>

      <View className='module-grid'>
        {/* 用户管理 */}
        {(
          <View className='module-card' onClick={navigateToUserManagement}>
            <View className='module-icon'>
              <User size={32} color='#1890ff' />
            </View>
            <View className='module-info'>
              <Text className='module-title'>用户管理</Text>
              <Text className='module-desc'>管理系统用户</Text>
            </View>
            <View className='module-arrow'>{'>'}</View>
          </View>
        )}

        {/* 角色管理 */}
        {(
          <View className='module-card' onClick={navigateToRoleManagement}>
            <View className='module-icon'>
              <ShieldOutlined size={32} color='#722ed1' />
            </View>
            <View className='module-info'>
              <Text className='module-title'>角色管理</Text>
              <Text className='module-desc'>管理角色权限</Text>
            </View>
            <View className='module-arrow'>{'>'}</View>
          </View>
        )}

        {/* 系统设置 */}
        <View className='module-card disabled'>
          <View className='module-icon'>
            <SettingOutlined size={32} color='#8c8c8c' />
          </View>
          <View className='module-info'>
            <Text className='module-title'>系统设置</Text>
            <Text className='module-desc'>配置系统参数</Text>
          </View>
          <View className='module-arrow'>{'>'}</View>
        </View>
      </View>

      <View className='dashboard-footer'>
        <Text className='footer-text'>
          Dehaze System Management v1.0
        </Text>
      </View>
    </View>
  );
};

export default Dashboard;
