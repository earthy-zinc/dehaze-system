import {Text, View} from '@tarojs/components';
import React from 'react';
import './index.less';

const Dashboard: React.FC = () => {
  return (
    <View className='dashboard-container'>
      <Text className='welcome-text'>欢迎来到 Dashboard 页面</Text>
    </View>
  );
};

export default Dashboard;
