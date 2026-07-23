/**
 * 图片预览组件
 */

import React, { useRef } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Animated,
} from 'react-native';
import ImageLoader from '@/components/ImageLoader';
import Icon from '@/components/Icon';
import Card from '@/components/Card';
import Button from '@/components/Button';
import { useResponsive } from '@/hooks/useResponsive';
import { theme } from '@/theme';
import type { SelectedImage } from '@/types/image';
import { formatFileSize } from '@/utils/file';

interface ImagePreviewProps {
  image: SelectedImage;
  onRemove: () => void;
  onNext: () => void;
}

/** 根据来源返回标签文案 */
function getSourceLabel(source?: string): string | null {
  switch (source) {
    case 'upload':
      return '上传图片';
    case 'camera':
      return '拍照图片';
    case 'sample':
      return '样例图片';
    case 'history':
      return '历史图片';
    default:
      return null;
  }
}

const ImagePreview: React.FC<ImagePreviewProps> = ({
  image,
  onRemove,
  onNext,
}) => {
  const { isMobile } = useResponsive();
  const fadeAnim = useRef(new Animated.Value(0)).current;

  // 入场动画
  React.useEffect(() => {
    Animated.timing(fadeAnim, {
      toValue: 1,
      duration: 300,
      useNativeDriver: true,
    }).start();
  }, [fadeAnim]);

  // 计算预览区域高度
  const previewHeight = isMobile ? 280 : 350;

  // 格式化文件大小
  const formattedSize = formatFileSize(image.size || 0);

  // 格式化尺寸
  const formattedDimensions = `${image.width || 0} × ${image.height || 0}`;

  // 来源标签
  const sourceLabel = getSourceLabel(image.source);

  return (
    <Animated.View style={[styles.container, { opacity: fadeAnim }]}>
      <Card padding={theme.spacing.lg} elevation>
        {/* 头部 */}
        <View style={styles.header}>
          <Text style={styles.title}>图片预览</Text>
          <TouchableOpacity
            onPress={onRemove}
            style={styles.removeButton}
            hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
          >
            <Icon name="close" size={20} color={theme.colors.status.error} />
          </TouchableOpacity>
        </View>

        {/* 图片预览区域 */}
        <View style={[styles.imageContainer, { height: previewHeight }]}>
          <ImageLoader
            source={{ uri: image.url }}
            style={styles.image}
            resizeMode="contain"
          />

          {/* 来源标签 */}
          {sourceLabel && (
            <View style={styles.sampleBadge}>
              <Icon name="images" size={12} color="#fff" />
              <Text style={styles.sampleBadgeText}>{sourceLabel}</Text>
            </View>
          )}
        </View>

        {/* 文件信息 */}
        <View style={styles.infoContainer}>
          <View style={styles.infoItem}>
            <Icon name="document" size={16} color={theme.colors.primary} />
            <Text style={styles.infoText} numberOfLines={1}>
              {image.name || '未命名'}
            </Text>
          </View>

          <View style={styles.infoRow}>
            <View style={styles.infoItem}>
              <Icon name="resize" size={16} color={theme.colors.status.success} />
              <Text style={styles.infoText}>{formattedDimensions}</Text>
            </View>

            <View style={styles.infoItem}>
              <Icon name="cloud" size={16} color={theme.colors.status.info} />
              <Text style={styles.infoText}>{formattedSize}</Text>
            </View>
          </View>
        </View>

        {/* 样例图片额外信息 */}
        {image.sampleInfo?.sceneType && (
          <View style={styles.sampleInfoContainer}>
            <View style={styles.sampleInfoItem}>
              <Text style={styles.sampleInfoLabel}>场景类型</Text>
              <Text style={styles.sampleInfoValue}>{image.sampleInfo.sceneType}</Text>
            </View>
          </View>
        )}

        {/* 下一步按钮 */}
        <View style={styles.buttonContainer}>
          <Button
            title="下一步：选择算法"
            onPress={onNext}
            variant="primary"
            icon={<Icon name="arrow-forward" size={18} color="#fff" />}
          />
        </View>
      </Card>
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  container: {
    marginTop: theme.spacing.lg,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: theme.spacing.md,
  },
  title: {
    fontSize: theme.typography.sizes.h5,
    fontWeight: theme.typography.weights.semibold,
    color: theme.colors.text.primary,
  },
  removeButton: {
    padding: theme.spacing.xs,
  },
  imageContainer: {
    backgroundColor: theme.colors.background.secondary,
    borderRadius: theme.layout.borderRadius.lg,
    overflow: 'hidden',
    marginBottom: theme.spacing.md,
  },
  image: {
    width: '100%',
    height: '100%',
  },
  sampleBadge: {
    position: 'absolute',
    top: theme.spacing.sm,
    left: theme.spacing.sm,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: theme.colors.primary,
    paddingHorizontal: theme.spacing.sm,
    paddingVertical: 4,
    borderRadius: theme.layout.borderRadius.sm,
    gap: 4,
  },
  sampleBadgeText: {
    fontSize: theme.typography.sizes.caption,
    color: '#fff',
    fontWeight: theme.typography.weights.medium,
  },
  infoContainer: {
    marginBottom: theme.spacing.md,
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: theme.spacing.sm,
  },
  infoItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    flex: 1,
  },
  infoText: {
    fontSize: theme.typography.sizes.body,
    color: theme.colors.text.secondary,
    flex: 1,
  },
  sampleInfoContainer: {
    flexDirection: 'row',
    backgroundColor: theme.colors.background.secondary,
    borderRadius: theme.layout.borderRadius.md,
    padding: theme.spacing.md,
    marginBottom: theme.spacing.md,
    gap: theme.spacing.lg,
  },
  sampleInfoItem: {
    flex: 1,
  },
  sampleInfoLabel: {
    fontSize: theme.typography.sizes.caption,
    color: theme.colors.text.tertiary,
    marginBottom: 4,
  },
  sampleInfoValue: {
    fontSize: theme.typography.sizes.body,
    fontWeight: theme.typography.weights.medium,
    color: theme.colors.text.primary,
  },
  buttonContainer: {
    marginTop: theme.spacing.sm,
  },
});

export default ImagePreview;
