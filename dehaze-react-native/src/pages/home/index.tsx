import type { RootStackParamList } from '@/routes/types';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import React, { useEffect } from 'react';
import { StyleSheet, ScrollView, Alert } from 'react-native';
import { MainLayout } from '@/layout';
import { useAuth } from '@/store';
import HeroSection from './components/HeroSection';
import ShowcaseSection from './components/ShowcaseSection';
import FeaturesSection from './components/FeaturesSection';
import AlgorithmSection from './components/AlgorithmSection';
import TechSpecsSection from './components/TechSpecsSection';
import FinalCTASection from './components/FinalCTASection';

type HomeScreenProps = NativeStackScreenProps<RootStackParamList, 'Home'>;

const HomeScreen: React.FC<HomeScreenProps> = ({ navigation }) => {
  const { state, refreshUserInfo } = useAuth();

  // 首页加载时获取用户信息（登录后首次进入时 state.userInfo 可能为 null）
  useEffect(() => {
    if (!state.userInfo) {
      refreshUserInfo().catch(() => {
        // 获取失败不阻塞首页展示
      });
    }
  }, [state.userInfo, refreshUserInfo]);

  // 导航处理函数
  const handleStartPress = () => {
    navigation.navigate('ImageInput');
  };

  const handleDatasetPress = () => {
    navigation.navigate('Dataset');
  };

  const handleImageInputPress = () => {
    navigation.navigate('ImageInput');
  };

  const handleAlgorithmSelectPress = () => {
    navigation.navigate('AlgorithmSelect');
  };

  const handleProcessingPress = () => {
    Alert.alert(
      '提示',
      '请先选择图片和算法后再进行去雾处理',
      [{ text: '去选择图片', onPress: () => navigation.navigate('ImageInput') }, { text: '取消' }]
    );
  };

  const handleComparePress = () => {
    Alert.alert(
      '提示',
      '请先完成去雾处理后才能使用效果对比功能',
      [{ text: '去处理', onPress: () => navigation.navigate('ImageInput') }, { text: '取消' }]
    );
  };

  const handleSideBySidePress = handleComparePress;
  const handleOverlayPress = handleComparePress;
  const handleMagnifierPress = handleComparePress;
  const handleFilterPress = handleComparePress;
  const handleMetricsPress = handleComparePress;

  const handleDatasetManagePress = () => {
    navigation.navigate('Dataset');
  };

  const handleTaskCenterPress = () => {
    navigation.navigate('Task');
  };

  const handleLearnMorePress = () => {
    navigation.navigate('Algorithm');
  };

  return (
    <MainLayout title="图像去雾系统">
      <ScrollView
        style={styles.scrollView}
        showsVerticalScrollIndicator={false}
        contentContainerStyle={styles.scrollContent}
      >
        <HeroSection
          onStartPress={handleStartPress}
          onDatasetPress={handleDatasetPress}
        />

        <ShowcaseSection onPress={handleStartPress} />

        <FeaturesSection
          onImageInputPress={handleImageInputPress}
          onAlgorithmSelectPress={handleAlgorithmSelectPress}
          onProcessingPress={handleProcessingPress}
          onComparePress={handleComparePress}
          onDatasetManagePress={handleDatasetManagePress}
          onTaskCenterPress={handleTaskCenterPress}
        />

        <AlgorithmSection onLearnMorePress={handleLearnMorePress} />

        <TechSpecsSection />

        <FinalCTASection onStartPress={handleStartPress} />
      </ScrollView>
    </MainLayout>
  );
};

const styles = StyleSheet.create({
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
  },
});

export default HomeScreen;
