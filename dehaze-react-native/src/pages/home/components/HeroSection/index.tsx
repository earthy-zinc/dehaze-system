import React from 'react';
import { View, Text, StyleSheet, ScrollView, Dimensions } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import Button from '@/components/Button';
import Icon from '@/components/Icon';

interface HeroSectionProps {
  onStartPress: () => void;
  onDatasetPress: () => void;
}

const { height } = Dimensions.get('window');

const HeroSection: React.FC<HeroSectionProps> = ({
  onStartPress,
  onDatasetPress,
}) => {
  return (
    <SafeAreaView style={styles.container}>
      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.heroContent}>
          <Text style={styles.heroTitle}>图像去雾</Text>
          <Text style={styles.heroSubtitle}>专业级图像处理系统</Text>
          <Text style={styles.heroDescription}>
            采用先进的深度学习算法，一键还原清晰视界{'\n'}
            从图像输入到效果评估的完整闭环体验
          </Text>

          <View style={styles.ctaContainer}>
            <Button
              title="立即开始"
              onPress={onStartPress}
              variant="primary"
              icon={<Icon name="arrow-right" size={16} color="#ffffff" />}
              style={styles.ctaButton}
            />
            <Button
              title="浏览数据集"
              onPress={onDatasetPress}
              variant="secondary"
              style={styles.ctaButton}
            />
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#ffffff',
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 60,
    minHeight: height,
  },
  heroContent: {
    alignItems: 'center',
    width: '100%',
    maxWidth: 600,
  },
  heroTitle: {
    fontSize: 48,
    fontWeight: '700',
    letterSpacing: -1,
    color: '#1e40af',
    marginBottom: 8,
    textAlign: 'center',
    lineHeight: 52.8,
  },
  heroSubtitle: {
    fontSize: 28,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 20,
    letterSpacing: -0.5,
    textAlign: 'center',
    lineHeight: 33.6,
  },
  heroDescription: {
    fontSize: 18,
    color: '#6b7280',
    lineHeight: 28.8,
    textAlign: 'center',
    marginBottom: 40,
    maxWidth: 600,
  },
  ctaContainer: {
    flexDirection: 'row',
    gap: 16,
    justifyContent: 'center',
    flexWrap: 'wrap',
    width: '100%',
  },
  ctaButton: {
    minWidth: 140,
  },
});

export default HeroSection;