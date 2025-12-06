import React from 'react';
import { View, StyleSheet } from 'react-native';
import Section from '@/components/Section';
import SpecCard from './SpecCard';
import { useResponsive } from '@/hooks/useResponsive';

const TechSpecsSection: React.FC = () => {
  const { width, isMobile, isTablet, spacing, containerPadding } = useResponsive();

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

  // 响应式列数计算
  const columns = isMobile ? 2 : isTablet ? 2 : 4;
  const cardWidth = (width - containerPadding * 2 - spacing * (columns - 1)) / columns;

  return (
    <Section
      title={undefined}
      subtitle={undefined}
      padding={isMobile ? 60 : 80}
      style={styles.container}
    >
      <View style={[
        styles.specsGrid,
        { paddingHorizontal: containerPadding, gap: spacing },
      ]}>
        {specs.map((spec, index) => (
          <View 
            key={`spec-${index}`} 
            style={[
              styles.specWrapper,
              { width: cardWidth },
            ]}
          >
            <SpecCard {...spec} compact={isMobile} />
          </View>
        ))}
      </View>
    </Section>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#ffffff',
  },
  specsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  specWrapper: {
    marginBottom: 16,
  },
});

export default TechSpecsSection;