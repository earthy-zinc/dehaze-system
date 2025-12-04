import React from 'react';
import { View, StyleSheet, ScrollView, Dimensions } from 'react-native';
import Section from '@/components/Section';
import SpecCard from './SpecCard';

const { width } = Dimensions.get('window');

const TechSpecsSection: React.FC = () => {
  const specs = [
    {
      icon: 'bolt',
      title: '高性能',
      value: '60fps',
      description: '流畅运行，响应时间<200ms',
    },
    {
      icon: 'mobile-alt',
      title: '全平台',
      value: '100%',
      description: '完美适配手机、平板、桌面',
    },
    {
      icon: 'brain',
      title: '智能算法',
      value: '8+',
      description: '支持多种先进去雾算法',
    },
    {
      icon: 'chart-bar',
      title: '专业评估',
      value: '5+',
      description: '多维度定量分析指标',
    },
  ];

  return (
    <Section
      title={undefined}
      subtitle={undefined}
      padding={80}
      style={styles.container}
    >
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.specsContainer}
      >
        {specs.map((spec, index) => (
          <View key={`spec-${index}`} style={styles.specWrapper}>
            <SpecCard {...spec} />
          </View>
        ))}
      </ScrollView>
    </Section>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#ffffff',
  },
  specsContainer: {
    paddingHorizontal: 20,
    gap: 32,
  },
  specWrapper: {
    width: width - 40,
  },
});

export default TechSpecsSection;