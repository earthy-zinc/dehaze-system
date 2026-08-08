import React from 'react';
import { View, Text, StyleSheet, Animated, TouchableOpacity } from 'react-native';
import LinearGradient from 'react-native-linear-gradient';
import Ionicons from 'react-native-vector-icons/Ionicons';
import { useResponsive } from '@/hooks/useResponsive';
import { useFadeSlideAnimation } from '@/hooks/useAnimation';
import { theme } from '@/theme';
import { gradientColors } from '@/theme/colors';

interface HeroSectionProps {
  onStartPress: () => void;
}

const HeroSection: React.FC<HeroSectionProps> = ({ onStartPress }) => {
  const { isMobile, fontScale, containerPadding, height } = useResponsive();
  
  const { animatedStyle } = useFadeSlideAnimation({
    scale: { initial: 0.95, final: 1 },
    slideDistance: 30,
  });

  // 响应式字体大小
  const titleFontSize = isMobile ? 40 : 48 * fontScale;
  const subtitleFontSize = isMobile ? 18 : 22 * fontScale;

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
          AI 图像去雾
        </Text>
        <Text
          style={[
            styles.heroSubtitle,
            { fontSize: subtitleFontSize, lineHeight: subtitleFontSize * theme.typography.lineHeights.title },
          ]}
        >
          一键清晰，还原真实视界
        </Text>

        <TouchableOpacity
          activeOpacity={0.85}
          onPress={onStartPress}
          style={styles.ctaWrapper}
        >
          <LinearGradient
            colors={gradientColors.primary}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
            style={styles.ctaButton}
          >
            <Ionicons name="flash" size={20} color="#fff" />
            <Text style={styles.ctaText}>开始去雾</Text>
          </LinearGradient>
        </TouchableOpacity>
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
    fontWeight: theme.typography.weights.regular,
    color: theme.colors.text.secondary,
    marginBottom: theme.spacing.xl,
    letterSpacing: theme.typography.letterSpacing.normal,
    textAlign: 'center',
  },
  ctaWrapper: {
    width: '100%',
    maxWidth: 320,
    borderRadius: theme.layout.borderRadius.full,
    ...theme.layout.shadows.lg,
  },
  ctaButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 14,
    paddingHorizontal: theme.spacing.xl,
    borderRadius: theme.layout.borderRadius.full,
    gap: 8,
  },
  ctaText: {
    fontSize: 18,
    fontWeight: '700',
    color: '#fff',
  },
});

export default HeroSection;