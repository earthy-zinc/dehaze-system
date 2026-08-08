import React, { useEffect } from 'react';
import { StyleSheet, ScrollView, Alert, View, Text, TouchableOpacity, Dimensions } from 'react-native';
import Ionicons from 'react-native-vector-icons/Ionicons';
import { useAuthStore } from '@/store';
import { colors } from '@/theme/colors';
import { spacing, layout } from '@/theme/spacing';
import HeroSection from './components/HeroSection';
import ShowcaseSection from './components/ShowcaseSection';
import FeaturesSection from './components/FeaturesSection';
import AlgorithmSection from './components/AlgorithmSection';
import TechSpecsSection from './components/TechSpecsSection';
import FinalCTASection from './components/FinalCTASection';
import type { BottomTabScreenProps } from '@react-navigation/bottom-tabs';
import type { TabParamList } from '@/routes/types';

const { width: screenWidth } = Dimensions.get('window');

type HomeScreenProps = BottomTabScreenProps<TabParamList, 'Home'>;

const HomeScreen: React.FC<HomeScreenProps> = ({ navigation }) => {
  const userInfo = useAuthStore(s => s.userInfo);
  const refreshUserInfo = useAuthStore(s => s.refreshUserInfo);

  useEffect(() => {
    if (!userInfo) {
      refreshUserInfo().catch(() => {
        // 获取失败不阻塞首页展示
      });
    }
  }, [userInfo, refreshUserInfo]);

  // "开始去雾"：跳转到去雾 Tab
  const handleStartDehaze = () => {
    navigation.jumpTo('Dehaze' as any);
  };

  // 跨 Stack 导航用 getParent()（arch-infra 导航重构后，Tab 内嵌套 Stack）
  const nav = navigation.getParent() ?? navigation;

  const handleImageInputPress = () => {
    (nav as any).navigate('ImageInput');
  };

  const handleAlgorithmSelectPress = () => {
    (nav as any).navigate('AlgorithmSelect');
  };

  const handleProcessingPress = () => {
    Alert.alert(
      '提示',
      '请先选择图片和算法后再进行去雾处理',
      [{ text: '去选择图片', onPress: () => (nav as any).navigate('ImageInput') }, { text: '取消' }]
    );
  };

  const handleComparePress = () => {
    Alert.alert(
      '提示',
      '请先完成去雾处理后才能使用效果对比功能',
      [{ text: '去处理', onPress: () => (nav as any).navigate('ImageInput') }, { text: '取消' }]
    );
  };

  const handleDatasetManagePress = () => {
    (nav as any).navigate('Dataset');
  };

  const handleTaskCenterPress = () => {
    (nav as any).navigate('Task');
  };

  const handleLearnMorePress = () => {
    (nav as any).navigate('AlgorithmBrowse');
  };

  return (
    <ScrollView
      style={styles.scrollView}
      showsVerticalScrollIndicator={false}
      contentContainerStyle={styles.scrollContent}
    >
      <HeroSection
        onStartPress={handleStartDehaze}
      />

      {/* 快捷入口 */}
      <View style={styles.quickEntrySection}>
        {[
          { icon: 'flash', label: '快速体验', route: 'Dehaze', color: '#3b82f6' },
          { icon: 'time', label: '处理历史', route: 'Task', color: '#6366f1' },
          { icon: 'images', label: '样例库', route: 'ImageInput', color: '#8b5cf6' },
          { icon: 'diamond', label: '会员权益', route: 'PersonalMember', color: '#f59e0b' },
        ].map((item, i) => (
          <TouchableOpacity
            key={i}
            style={styles.quickEntryCard}
            onPress={() => {
              if (item.route === 'Dehaze') {
                navigation.jumpTo('Dehaze' as any);
              } else {
                (nav as any).navigate(item.route);
              }
            }}
          >
            <View style={[styles.quickEntryIcon, { backgroundColor: item.color + '15' }]}>
              <Ionicons name={item.icon} size={24} color={item.color} />
            </View>
            <Text style={styles.quickEntryLabel}>{item.label}</Text>
          </TouchableOpacity>
        ))}
      </View>

      {/* 数据统计 */}
      <View style={styles.statsSection}>
        {[
          { label: '算法数量', value: '20+' },
          { label: '处理张数', value: '10M+' },
          { label: '用户评分', value: '4.9' },
        ].map((item, i) => (
          <View key={i} style={styles.statItem}>
            <Text style={styles.statValue}>{item.value}</Text>
            <Text style={styles.statLabel}>{item.label}</Text>
          </View>
        ))}
      </View>

      <ShowcaseSection onPress={handleStartDehaze} />

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

      <FinalCTASection onStartPress={handleStartDehaze} />
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  scrollView: {
    flex: 1,
    backgroundColor: colors.background.secondary,
  },
  scrollContent: {
    flexGrow: 1,
  },
  quickEntrySection: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
    paddingHorizontal: spacing.lg,
    marginTop: spacing.lg,
  },
  quickEntryCard: {
    width: (screenWidth - spacing.lg * 2 - spacing.sm) / 2,
    backgroundColor: colors.background.primary,
    borderRadius: layout.borderRadius.xl,
    padding: spacing.md,
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    ...layout.shadows.sm,
    borderWidth: 1,
    borderColor: colors.border.light,
  },
  quickEntryIcon: {
    width: 44,
    height: 44,
    borderRadius: layout.borderRadius.lg,
    justifyContent: 'center',
    alignItems: 'center',
  },
  quickEntryLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.text.primary,
  },
  statsSection: {
    flexDirection: 'row',
    backgroundColor: colors.background.primary,
    borderRadius: layout.borderRadius.xl,
    padding: spacing.lg,
    marginHorizontal: spacing.lg,
    marginTop: spacing.md,
    borderWidth: 1,
    borderColor: colors.border.light,
    ...layout.shadows.sm,
  },
  statItem: {
    flex: 1,
    alignItems: 'center',
  },
  statValue: {
    fontSize: 24,
    fontWeight: '700',
    color: colors.primary,
  },
  statLabel: {
    fontSize: 12,
    color: colors.text.secondary,
    marginTop: 4,
  },
});

export default HomeScreen;
