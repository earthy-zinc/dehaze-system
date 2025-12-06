import React from 'react';
import { View, Text, StyleSheet, Animated } from 'react-native';
import Button from '@/components/Button';
import Icon from '@/components/Icon';
import { useResponsive } from '@/hooks/useResponsive';
import { useFadeSlideAnimation } from '@/hooks/useAnimation';
import { theme } from '@/theme';

interface HeroSectionProps {
  onStartPress: () => void;
  onDatasetPress: () => void;
}

const HeroSection: React.FC<HeroSectionProps> = ({
  onStartPress,
  onDatasetPress,
}) => {
  const { isMobile, fontScale, containerPadding, height } = useResponsive();
  
  const { animatedStyle } = useFadeSlideAnimation({
    scale: { initial: 0.95, final: 1 },
    slideDistance: 30,
  });

  // 响应式字体大小
  const titleFontSize = isMobile ? 36 : 48 * fontScale;
  const subtitleFontSize = isMobile ? 22 : 28 * fontScale;
  const descFontSize = isMobile ? 16 : 18 * fontScale;

  return (
    <View style={[styles.container, { minHeight: height * 0.85 }]}>
      <Animated.View
        style={[
          styles.heroContent,
          animatedStyle,
          { paddingHorizontal: containerPadding },
        ]}
      >
        <Text
          style={[
            styles.heroTitle,
            { fontSize: titleFontSize, lineHeight: titleFontSize * theme.typography.lineHeights.hero },
          ]}
        >
          图像去雾
        </Text>
        <Text
          style={[
            styles.heroSubtitle,
            { fontSize: subtitleFontSize, lineHeight: subtitleFontSize * theme.typography.lineHeights.title },
          ]}
        >
          专业级图像处理系统
        </Text>
        <Text
          style={[
            styles.heroDescription,
            { fontSize: descFontSize, lineHeight: descFontSize * theme.typography.lineHeights.body },
          ]}
        >
          采用先进的深度学习算法，一键还原清晰视界{'\n'}
          从图像输入到效果评估的完整闭环体验
        </Text>

        <View style={[
          styles.ctaContainer,
          isMobile ? styles.ctaContainerMobile : null,
        ]}>
          <Button
            title="立即开始"
            onPress={onStartPress}
            variant="primary"
            icon={<Icon name="arrow-right" size={16} color={theme.colors.text.inverse} />}
            style={isMobile ? styles.ctaButtonMobile : styles.ctaButton}
          />
          <Button
            title="浏览数据集"
            onPress={onDatasetPress}
            variant="secondary"
            style={isMobile ? styles.ctaButtonMobile : styles.ctaButton}
          />
        </View>
      </Animated.View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: theme.colors.background.primary,
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: theme.spacing.huge,
  },
  heroContent: {
    alignItems: 'center',
    width: '100%',
    maxWidth: 600,
  },
  heroTitle: {
    fontWeight: theme.typography.weights.bold,
    letterSpacing: theme.typography.letterSpacing.tight,
    color: theme.colors.primaryDark,
    marginBottom: theme.spacing.sm,
    textAlign: 'center',
  },
  heroSubtitle: {
    fontWeight: theme.typography.weights.semibold,
    color: theme.colors.text.primary,
    marginBottom: theme.spacing.lg,
    letterSpacing: theme.typography.letterSpacing.normal,
    textAlign: 'center',
  },
  heroDescription: {
    color: theme.colors.text.secondary,
    textAlign: 'center',
    marginBottom: theme.spacing.xxl,
    maxWidth: 600,
  },
  ctaContainer: {
    flexDirection: 'row',
    gap: theme.spacing.md,
    justifyContent: 'center',
    flexWrap: 'wrap',
    width: '100%',
  },
  ctaContainerMobile: {
    flexDirection: 'column',
    alignItems: 'center',
  },
  ctaButton: {
    minWidth: 160,
  },
  ctaButtonMobile: {
    width: '100%',
    maxWidth: 280,
  },
});

export default HeroSection;