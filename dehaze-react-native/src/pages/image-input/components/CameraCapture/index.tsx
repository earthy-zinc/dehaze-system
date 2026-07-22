/**
 * 拍照组件
 */

import React, { useCallback, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Alert,
  Linking,
} from 'react-native';
import { launchCamera, ImagePickerResponse } from 'react-native-image-picker';
import Icon from '@/components/Icon';
import Button from '@/components/Button';
import { theme } from '@/theme';
import type { SelectedImage } from '@/types/image';
import { imageInputApi } from '../../services/imageInputApi';

interface CameraCaptureProps {
  onCapture: (image: SelectedImage) => void;
  loading?: boolean;
}

const CameraCapture: React.FC<CameraCaptureProps> = ({
  onCapture,
  loading = false,
}) => {
  const [uploading, setUploading] = useState(false);
  const busy = loading || uploading;

  const handleOpenCamera = useCallback(async () => {
    if (busy) return;

    try {
      const result: ImagePickerResponse = await launchCamera({
        mediaType: 'photo',
        quality: 0.9,
        includeBase64: false,
        saveToPhotos: false,
        cameraType: 'back',
      });

      if (result.didCancel) {
        return;
      }

      if (result.errorCode) {
        if (result.errorCode === 'camera_unavailable') {
          Alert.alert('相机不可用', '请检查相机权限设置');
        } else if (result.errorCode === 'permission') {
          Alert.alert(
            '需要相机权限',
            '请在设置中允许应用访问相机',
            [
              { text: '取消', style: 'cancel' },
              { text: '去设置', onPress: () => Linking.openSettings() },
            ]
          );
        } else {
          Alert.alert('错误', result.errorMessage || '拍照失败');
        }
        return;
      }

      const asset = result.assets?.[0];
      if (!asset || !asset.uri) {
        Alert.alert('错误', '无法获取图片信息');
        return;
      }

      // 获取图片尺寸
      let width = asset.width || 0;
      let height = asset.height || 0;

      if (!width || !height) {
        try {
          const size = await imageInputApi.getImageSize(asset.uri);
          width = size.width;
          height = size.height;
        } catch (e) {
          width = 1920;
          height = 1080;
        }
      }

      const fileName = asset.fileName || `photo_${Date.now()}.jpg`;

      // 上传到后端文件服务，获取后端可访问的远程 URL
      setUploading(true);
      let fileInfo;
      try {
        fileInfo = await imageInputApi.uploadImage(asset.uri, fileName, asset.type);
      } catch (error) {
        setUploading(false);
        Alert.alert('上传失败', error instanceof Error ? error.message : '图片上传失败，请重试');
        return;
      }
      setUploading(false);

      const capturedImage: SelectedImage = {
        id: Date.now().toString(),
        url: fileInfo.url,
        thumbUrl: asset.uri,
        name: fileInfo.name || fileName,
        width,
        height,
        size: asset.fileSize || 0,
        source: 'camera',
      };

      onCapture(capturedImage);
    } catch (error) {
      console.warn('Camera error:', error);
      Alert.alert('错误', '打开相机时发生错误');
    }
  }, [busy, onCapture]);

  return (
    <View style={styles.container}>
      <View style={styles.iconContainer}>
        <Icon name="camera" size={64} color={theme.colors.text.tertiary} />
      </View>

      <Text style={styles.title}>使用相机拍照</Text>
      <Text style={styles.description}>
        点击下方按钮打开相机，拍摄需要去雾的图片
      </Text>

      <View style={styles.buttonContainer}>
        <Button
          title="打开相机"
          onPress={handleOpenCamera}
          variant="primary"
          loading={busy}
          icon={<Icon name="camera" size={18} color="#fff" />}
        />
      </View>

      <View style={styles.tipsContainer}>
        <View style={styles.tipItem}>
          <Icon name="checkmark-circle" size={16} color={theme.colors.status.success} />
          <Text style={styles.tipText}>支持前后摄像头切换</Text>
        </View>
        <View style={styles.tipItem}>
          <Icon name="checkmark-circle" size={16} color={theme.colors.status.success} />
          <Text style={styles.tipText}>自动保存高质量照片</Text>
        </View>
        <View style={styles.tipItem}>
          <Icon name="information-circle" size={16} color={theme.colors.status.info} />
          <Text style={styles.tipText}>建议在光线充足的环境下拍摄</Text>
        </View>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: theme.colors.background.secondary,
    borderRadius: theme.layout.borderRadius.xl,
    padding: theme.spacing.xxl,
    alignItems: 'center',
  },
  iconContainer: {
    width: 120,
    height: 120,
    borderRadius: 60,
    backgroundColor: theme.colors.background.primary,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: theme.spacing.xl,
  },
  title: {
    fontSize: theme.typography.sizes.h5,
    fontWeight: theme.typography.weights.semibold,
    color: theme.colors.text.primary,
    marginBottom: theme.spacing.sm,
  },
  description: {
    fontSize: theme.typography.sizes.body,
    color: theme.colors.text.secondary,
    textAlign: 'center',
    marginBottom: theme.spacing.xl,
    lineHeight: 22,
  },
  buttonContainer: {
    width: '100%',
    maxWidth: 200,
    marginBottom: theme.spacing.xl,
  },
  tipsContainer: {
    width: '100%',
    gap: theme.spacing.sm,
  },
  tipItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.sm,
  },
  tipText: {
    fontSize: theme.typography.sizes.caption,
    color: theme.colors.text.secondary,
  },
});

export default CameraCapture;
