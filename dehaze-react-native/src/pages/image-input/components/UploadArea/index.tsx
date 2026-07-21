/**
 * 图片上传区域组件
 */

import React, { useRef, useCallback } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Animated,
  ActivityIndicator,
  Alert,
} from 'react-native';
import { launchImageLibrary, ImagePickerResponse } from 'react-native-image-picker';
import Icon from '@/components/Icon';
import { theme } from '@/theme';
import type { SelectedImage } from '@/types/image';
import { imageInputApi } from '../../services/imageInputApi';

interface UploadAreaProps {
  onImageSelected: (image: SelectedImage) => void;
  loading?: boolean;
}

const UploadArea: React.FC<UploadAreaProps> = ({
  onImageSelected,
  loading = false,
}) => {
  const scaleAnim = useRef(new Animated.Value(1)).current;

  const handlePressIn = () => {
    Animated.spring(scaleAnim, {
      toValue: 0.98,
      useNativeDriver: true,
      tension: 100,
      friction: 8,
    }).start();
  };

  const handlePressOut = () => {
    Animated.spring(scaleAnim, {
      toValue: 1,
      useNativeDriver: true,
      tension: 100,
      friction: 8,
    }).start();
  };

  const handleSelectImage = useCallback(async () => {
    if (loading) return;

    try {
      const result: ImagePickerResponse = await launchImageLibrary({
        mediaType: 'photo',
        quality: 0.9,
        includeBase64: false,
        selectionLimit: 1,
      });

      if (result.didCancel) {
        return;
      }

      if (result.errorCode) {
        Alert.alert('错误', result.errorMessage || '选择图片失败');
        return;
      }

      const asset = result.assets?.[0];
      if (!asset || !asset.uri) {
        Alert.alert('错误', '无法获取图片信息');
        return;
      }

      // 验证图片
      const validation = imageInputApi.validateImage(
        asset.fileSize || 0,
        asset.type
      );

      if (!validation.valid) {
        Alert.alert('图片不符合要求', validation.error);
        return;
      }

      // 检查是否需要压缩
      if (imageInputApi.needsCompression(asset.fileSize || 0)) {
        Alert.alert(
          '图片较大',
          '图片大于5MB，建议压缩后上传以获得更好的体验',
          [
            { text: '取消', style: 'cancel' },
            {
              text: '继续使用',
              onPress: () => processImage(asset),
            },
          ]
        );
        return;
      }

      processImage(asset);
    } catch (error) {
      console.warn('Image selection error:', error);
      Alert.alert('错误', '选择图片时发生错误');
    }
  }, [loading, onImageSelected]);

  const processImage = useCallback(async (asset: any) => {
    try {
      // 获取图片尺寸
      let width = asset.width || 0;
      let height = asset.height || 0;

      if (!width || !height) {
        try {
          const size = await imageInputApi.getImageSize(asset.uri);
          width = size.width;
          height = size.height;
        } catch (e) {
          // 使用默认值
          width = 1920;
          height = 1080;
        }
      }

      const selectedImage: SelectedImage = {
        id: Date.now().toString(),
        url: asset.uri,
        name: asset.fileName || `image_${Date.now()}.jpg`,
        width,
        height,
        size: asset.fileSize || 0,
        source: 'upload',
      };

      onImageSelected(selectedImage);
    } catch (error) {
      console.warn('Process image error:', error);
      Alert.alert('错误', '处理图片时发生错误');
    }
  }, [onImageSelected]);

  return (
    <TouchableOpacity
      onPress={handleSelectImage}
      onPressIn={handlePressIn}
      onPressOut={handlePressOut}
      activeOpacity={1}
      disabled={loading}
    >
      <Animated.View
        style={[
          styles.container,
          { transform: [{ scale: scaleAnim }] },
        ]}
      >
        {loading ? (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color={theme.colors.primary} />
            <Text style={styles.loadingText}>正在处理图片...</Text>
          </View>
        ) : (
          <>
            <View style={styles.iconContainer}>
              <Icon name="cloud-upload" size={48} color={theme.colors.primary} />
            </View>
            <Text style={styles.title}>点击选择图片</Text>
            <Text style={styles.subtitle}>
              支持 JPG、PNG、WEBP、HEIC 格式
            </Text>
            <Text style={styles.hint}>最大 20MB</Text>
          </>
        )}
      </Animated.View>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: theme.colors.background.secondary,
    borderRadius: theme.layout.borderRadius.xl,
    borderWidth: 2,
    borderColor: theme.colors.border.light,
    borderStyle: 'dashed',
    padding: theme.spacing.xxxl,
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 200,
  },
  iconContainer: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: `${theme.colors.primary}15`,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: theme.spacing.lg,
  },
  title: {
    fontSize: theme.typography.sizes.h5,
    fontWeight: theme.typography.weights.semibold,
    color: theme.colors.text.primary,
    marginBottom: theme.spacing.sm,
  },
  subtitle: {
    fontSize: theme.typography.sizes.body,
    color: theme.colors.text.secondary,
    marginBottom: theme.spacing.xs,
  },
  hint: {
    fontSize: theme.typography.sizes.caption,
    color: theme.colors.text.tertiary,
  },
  loadingContainer: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  loadingText: {
    fontSize: theme.typography.sizes.body,
    color: theme.colors.text.secondary,
    marginTop: theme.spacing.md,
  },
});

export default UploadArea;
