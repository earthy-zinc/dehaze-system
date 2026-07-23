import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, Animated } from 'react-native';
import Section from '@/components/Section';
import ImageLoader from '@/components/ImageLoader';
import Card from '@/components/Card';
import { useResponsive } from '@/hooks/useResponsive';
import { useFadeSlideAnimation } from '@/hooks/useAnimation';
import { theme } from '@/theme';
import { imageInputApi } from '@/pages/image-input/services/imageInputApi';

interface ShowcaseSectionProps {
  onPress?: () => void;
}

const ShowcaseSection: React.FC<ShowcaseSectionProps> = ({ onPress }) => {
  const { width, isMobile, containerPadding, fontScale } = useResponsive();
  
  const { animatedStyle } = useFadeSlideAnimation({
    scale: { initial: 0.9, final: 1 },
    slideDistance: 0, // Only scale and fade
  });

  // 从后端获取 NH-HAZE-2023 样张（由 nginx-dataset:9000 直服），避免硬编码外部图片 URL
  const [showcaseImageUrl, setShowcaseImageUrl] = useState<string>('');
  useEffect(() => {
    imageInputApi
      .getRandomSample()
      .then(sample => setShowcaseImageUrl(sample.url))
      .catch(() => {
        // 样例加载失败不阻塞页面，保留占位
      });
  }, []);

  // 响应式图片尺寸
  const imageWidth = width - containerPadding * 2;
  const imageHeight = isMobile ? 200 : Math.min(320, imageWidth * 0.5);

  return (
    <Section
      title="一键去雾，效果显著"
      subtitle="智能算法自动识别雾霾程度，精准还原图像细节"
      padding={isMobile ? theme.spacing.xxxl : theme.spacing.huge}
    >
      <Animated.View
        style={[
          styles.showcaseContainer,
          animatedStyle,
          { paddingHorizontal: containerPadding },
        ]}
      >
        <Card onPress={onPress} margin={0} padding={0} borderRadius={theme.layout.borderRadius.xxl}>
          <View style={styles.comparisonContainer}>
            {showcaseImageUrl ? (
              <ImageLoader
                source={{ uri: showcaseImageUrl }}
                style={styles.showcaseImage}
                containerStyle={{
                  ...styles.imageContainer,
                  width: imageWidth,
                  height: imageHeight,
                }}
              />
            ) : (
              <View
                style={{
                  ...styles.imageContainer,
                  width: imageWidth,
                  height: imageHeight,
                }}
              />
            )}
            <View style={[
              styles.comparisonLabel,
              isMobile && styles.comparisonLabelMobile,
            ]}>
              <View style={styles.labelItem}>
                <View style={[styles.labelDot, styles.beforeDot]} />
                <Text style={[
                  styles.labelText,
                  styles.beforeText,
                  isMobile ? styles.labelTextMobile : { fontSize: 15 * fontScale },
                ]}>
                  去雾前
                </Text>
              </View>
              <Text style={[styles.divider, isMobile ? styles.dividerMobile : undefined]}>→</Text>
              <View style={styles.labelItem}>
                <View style={[styles.labelDot, styles.afterDot]} />
                <Text style={[
                  styles.labelText,
                  styles.afterText,
                  isMobile ? styles.labelTextMobile : { fontSize: 15 * fontScale },
                ]}>
                  去雾后
                </Text>
              </View>
            </View>
          </View>
        </Card>
      </Animated.View>
    </Section>
  );
};

const styles = StyleSheet.create({
  showcaseContainer: {
    width: '100%',
  },
  comparisonContainer: {
    position: 'relative',
    overflow: 'hidden',
    borderRadius: theme.layout.borderRadius.xxl,
  },
  imageContainer: {
    minHeight: 200,
  },
  showcaseImage: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },
  comparisonLabel: {
    position: 'absolute',
    bottom: theme.spacing.lg,
    left: theme.spacing.lg,
    right: theme.spacing.lg,
    backgroundColor: theme.colors.background.overlay,
    paddingHorizontal: theme.spacing.lg,
    paddingVertical: theme.spacing.md,
    borderRadius: theme.layout.borderRadius.full,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: theme.spacing.md,
  },
  comparisonLabelMobile: {
    bottom: theme.spacing.md,
    left: theme.spacing.md,
    right: theme.spacing.md,
    paddingHorizontal: theme.spacing.md,
    paddingVertical: 10,
    gap: 12,
  },
  labelItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.sm,
  },
  labelDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  beforeDot: {
    backgroundColor: theme.colors.status.warning,
  },
  afterDot: {
    backgroundColor: theme.colors.status.success,
  },
  labelText: {
    fontWeight: theme.typography.weights.semibold,
    color: theme.colors.text.inverse,
  },
  beforeText: {
    color: theme.colors.status.warning,
  },
  afterText: {
    color: theme.colors.status.success,
  },
  divider: {
    fontSize: theme.typography.sizes.body,
    fontWeight: theme.typography.weights.semibold,
    color: theme.colors.text.muted,
  },
  dividerMobile: {
    fontSize: theme.typography.sizes.bodySmall,
  },
  labelTextMobile: {
    fontSize: theme.typography.sizes.caption,
  },
});

export default ShowcaseSection;