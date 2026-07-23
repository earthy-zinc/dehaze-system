import React from 'react';
import { View, Text, StyleSheet, Animated } from 'react-native';
import Button from '@/components/Button';
import Icon from '@/components/Icon';
import { useResponsive } from '@/hooks/useResponsive';
import { useFadeSlideAnimation } from '@/hooks/useAnimation';
import { theme } from '@/theme';

interface FinalCTASectionProps {
  onStartPress: () => void;
}

const FinalCTASection: React.FC<FinalCTASectionProps> = ({ onStartPress }) => {
  const { height, isMobile, fontScale, containerPadding } = useResponsive();

  const { animatedStyle } = useFadeSlideAnimation({ slideDistance: 30 });

  // 响应式字体大小
  const titleFontSize = isMobile ? theme.typography.sizes.h3 : theme.typography.sizes.h2 * fontScale;
  const subtitleFontSize = isMobile ? theme.typography.sizes.medium : theme.typography.sizes.large * fontScale;

  return (
    <View style={[
      styles.container,
      {
        paddingHorizontal: containerPadding,
        minHeight: height * 0.4,
      },
    ]}>
      <Animated.View style={[
        styles.content,
        animatedStyle,
      ]}>
        <Text style={[
          styles.ctaTitle,
          { fontSize: titleFontSize, lineHeight: titleFontSize * theme.typography.lineHeights.title },
        ]}>
          准备好体验专业级图像去雾了吗？
        </Text>
        <Text style={[
          styles.ctaSubtitle,
          { fontSize: subtitleFontSize, lineHeight: subtitleFontSize * theme.typography.lineHeights.body },
        ]}>
          立即开始，让您的图像重获清晰
        </Text>

        <Button
          title="开始使用"
          onPress={onStartPress}
          variant="large"
          icon={<Icon name="arrow-right" size={18} color={theme.colors.text.inverse} />}
          style={isMobile ? styles.ctaButtonMobile : styles.ctaButton}
        />
      </Animated.View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: theme.colors.background.primary,
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: theme.spacing.huge,
  },
  content: {
    alignItems: 'center',
    width: '100%',
    maxWidth: 600,
  },
  ctaTitle: {
    fontWeight: theme.typography.weights.bold,
    color: theme.colors.text.primary,
    marginBottom: theme.spacing.md,
    letterSpacing: theme.typography.letterSpacing.normal,
    textAlign: 'center',
  },
  ctaSubtitle: {
    color: theme.colors.text.secondary,
    marginBottom: theme.spacing.xxl,
    textAlign: 'center',
  },
  ctaButton: {
    width: 240,
    maxWidth: 280,
  },
  ctaButtonMobile: {
    width: '100%',
    maxWidth: 280,
  },
});

export default FinalCTASection;