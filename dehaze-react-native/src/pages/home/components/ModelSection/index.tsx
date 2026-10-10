import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, Animated } from 'react-native';
import ImageLoader from '@/components/ImageLoader';
import Icon from '@/components/Icon';
import Card from '@/components/Card';
import { useResponsive } from '@/hooks/useResponsive';
import { useFadeSlideAnimation } from '@/hooks/useAnimation';
import { theme } from '@/theme';
import { imageInputApi } from '@/pages/image-input/services/imageInputApi';

const ModelSection: React.FC = () => {
  const { isMobile, isTablet, fontScale, containerPadding } = useResponsive();

  // Text animation (slide in from left)
  const { animatedStyle: textAnimStyle } = useFadeSlideAnimation({
    direction: 'right', // Slide TO right (from left)
    slideDistance: 50,
  });

  // Image animation (slide in from right)
  const { animatedStyle: imageAnimStyle } = useFadeSlideAnimation({
    direction: 'left', // Slide TO right (from left)
    slideDistance: 50,
  });

  const modelFeatures = [
    '按场景智能选择最合适的去雾模型',
    '实时对比不同模型的处理效果',
    '毫秒级处理速度，即时查看结果',
    '支持批量处理和参数自定义',
  ];

  // 从后端获取 NH-HAZE-2023 样张（由 nginx-dataset:9000 直服），避免硬编码外部图片 URL
  const [modelImageUrl, setModelImageUrl] = useState<string>('');
  useEffect(() => {
    imageInputApi
      .getRandomSample()
      .then(sample => setModelImageUrl(sample.url))
      .catch(() => {
        // 样例加载失败不阻塞页面，保留占位
      });
  }, []);

  // 响应式字体大小
  const titleFontSize = isMobile
    ? theme.typography.sizes.h3
    : theme.typography.sizes.h1 * fontScale;
  const subtitleFontSize = isMobile
    ? theme.typography.sizes.medium
    : theme.typography.sizes.h6 * fontScale;

  // 响应式图片高度
  const imageHeight = isMobile ? 200 : isTablet ? 280 : 320;

  return (
    <View style={[styles.container, { paddingHorizontal: containerPadding }]}>
      <View style={[styles.modelContent, !isMobile && styles.modelContentRow]}>
        {/* Text Content */}
        <Animated.View
          style={[
            styles.modelText,
            !isMobile && styles.modelTextRow,
            textAnimStyle,
          ]}
        >
          <Text style={[styles.sectionTitle, { fontSize: titleFontSize }]}>
            多模型智能选择
          </Text>
          <Text
            style={[styles.sectionSubtitle, { fontSize: subtitleFontSize }]}
          >
            统一模型管理，按场景与速度档位挑选最合适的去雾模型
          </Text>

          <View style={styles.featuresList}>
            {modelFeatures.map((feature, index) => (
              <View key={`feature-${index}`} style={styles.featureItem}>
                <Icon
                  name="check-circle"
                  size={20}
                  color={theme.colors.status.success}
                  style={styles.featureIcon}
                />
                <Text style={styles.featureText}>{feature}</Text>
              </View>
            ))}
          </View>
        </Animated.View>

        {/* Image */}
        <Animated.View
          style={[
            styles.modelVisual,
            !isMobile && styles.modelVisualRow,
            imageAnimStyle,
          ]}
        >
          <Card
            padding={0}
            margin={0}
            borderRadius={theme.layout.borderRadius.xl}
          >
            {modelImageUrl ? (
              <ImageLoader
                source={{ uri: modelImageUrl }}
                style={styles.modelImage}
                containerStyle={{
                  ...styles.imageContainer,
                  height: imageHeight,
                }}
              />
            ) : (
              <View
                style={{
                  ...styles.imageContainer,
                  height: imageHeight,
                }}
              />
            )}
          </Card>
        </Animated.View>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    width: '100%',
    backgroundColor: theme.colors.primary,
    paddingVertical: theme.spacing.huge,
  },
  modelContent: {
    flexDirection: 'column',
    gap: theme.spacing.xxl,
  },
  modelContentRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.xxxl,
  },
  modelText: {
    flex: 1,
  },
  modelTextRow: {
    flex: 1,
  },
  sectionTitle: {
    fontWeight: theme.typography.weights.bold,
    color: theme.colors.text.inverse,
    marginBottom: theme.spacing.md,
    letterSpacing: theme.typography.letterSpacing.normal,
  },
  sectionSubtitle: {
    color: 'rgba(255, 255, 255, 0.8)',
    lineHeight: 28.8,
    marginBottom: theme.spacing.xl,
  },
  featuresList: {
    marginBottom: theme.spacing.xl,
  },
  featureItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: theme.spacing.md,
    marginBottom: theme.spacing.md,
  },
  featureIcon: {
    marginTop: 2,
  },
  featureText: {
    flex: 1,
    fontSize: theme.typography.sizes.medium,
    color: 'rgba(255, 255, 255, 0.95)',
    lineHeight:
      theme.typography.sizes.medium * theme.typography.lineHeights.body,
  },
  modelVisual: {
    flex: 1,
  },
  modelVisualRow: {
    flex: 1,
  },
  imageContainer: {
    width: '100%',
    borderRadius: theme.layout.borderRadius.xl,
    overflow: 'hidden',
  },
  modelImage: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },
});

export default ModelSection;
