/**
 * 图像输入页面
 */

import React, { useState, useCallback } from 'react';
import {
  View,
  ScrollView,
  StyleSheet,
  Alert,
} from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '@/routes/types';
import { MainLayout } from '@/layout';
import Card from '@/components/Card';
import { useResponsive } from '@/hooks/useResponsive';
import { theme } from '@/theme';
import type { SelectedImage } from '@/types/image';

// 类型
import { InputMethod } from './types/imageInput';

// 服务
import { imageInputApi } from './services/imageInputApi';

// 组件
import InputMethodSelector from './components/InputMethodSelector';
import UploadArea from './components/UploadArea';
import CameraCapture from './components/CameraCapture';
import SampleGallery from './components/SampleGallery';
import HistoryList from './components/HistoryList';
import ImagePreview from './components/ImagePreview';
import QuickStartBanner from './components/QuickStartBanner';

type Props = NativeStackScreenProps<RootStackParamList, 'ImageInput'>;

const ImageInputScreen: React.FC<Props> = ({ navigation }) => {
  const { containerPadding } = useResponsive();

  // 状态
  const [currentMethod, setCurrentMethod] = useState<InputMethod>('upload');
  const [selectedImage, setSelectedImage] = useState<SelectedImage | null>(null);
  const [loading, setLoading] = useState(false);

  // 处理输入方式切换
  const handleMethodChange = useCallback((method: InputMethod) => {
    setCurrentMethod(method);
  }, []);

  // 处理图片选择
  const handleImageSelected = useCallback((image: SelectedImage) => {
    setSelectedImage(image);
  }, []);

  // 处理移除图片
  const handleRemoveImage = useCallback(() => {
    setSelectedImage(null);
  }, []);

  // 处理下一步
  const handleNext = useCallback(() => {
    if (!selectedImage) {
      Alert.alert('提示', '请先选择一张图片');
      return;
    }

    // 导航到算法选择页面，传递选中的图片
    navigation.navigate('AlgorithmSelect', {
      image: selectedImage,
    });
  }, [selectedImage, navigation]);

  // 处理快速体验
  const handleQuickStart = useCallback(async () => {
    setLoading(true);
    try {
      // 随机选择一张样例图片
      const randomSample = await imageInputApi.getRandomSample();

      // 获取图片尺寸
      let width = randomSample.width || 1920;
      let height = randomSample.height || 1080;

      if (!width || !height) {
        try {
          const size = await imageInputApi.getImageSize(randomSample.url);
          width = size.width;
          height = size.height;
        } catch (e) {
          // 使用默认值
        }
      }

      const quickStartImage: SelectedImage = {
        id: randomSample.id.toString(),
        url: randomSample.url,
        thumbUrl: randomSample.thumbUrl,
        name: randomSample.name,
        width,
        height,
        source: 'sample',
        sampleInfo: {
          sceneType: randomSample.sceneType,
        },
      };

      setSelectedImage(quickStartImage);
      setCurrentMethod('sample');
    } catch (error) {
      Alert.alert('错误', error instanceof Error ? error.message : '加载样例图片失败');
    } finally {
      setLoading(false);
    }
  }, []);

  // 渲染当前输入方式的内容
  const renderInputContent = () => {
    switch (currentMethod) {
      case 'upload':
        return (
          <UploadArea
            onImageSelected={handleImageSelected}
            loading={loading}
          />
        );

      case 'camera':
        return (
          <CameraCapture
            onCapture={handleImageSelected}
            loading={loading}
          />
        );

      case 'sample':
        return (
          <SampleGallery
            onSelectSample={handleImageSelected}
          />
        );

      case 'history':
        return (
          <HistoryList
            onSelectRecord={handleImageSelected}
          />
        );

      default:
        return null;
    }
  };

  return (
    <MainLayout title="图像输入">
      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={[
          styles.scrollContent,
          { padding: containerPadding },
        ]}
        showsVerticalScrollIndicator={false}
      >
        {/* 主卡片 */}
        <Card elevation padding={theme.spacing.lg}>
          {/* 标题区域 */}
          <View style={styles.header}>
            <View style={styles.titleContainer}>
              <View style={styles.titleIcon}>
                <View style={styles.titleIconInner} />
              </View>
              <View>
                <View style={styles.titleRow}>
                  <View style={styles.titleDot} />
                  <View style={styles.titleTextContainer}>
                    <View style={styles.titleLine} />
                  </View>
                </View>
              </View>
            </View>
          </View>

          {/* 输入方式选择器 */}
          <InputMethodSelector
            currentMethod={currentMethod}
            onMethodChange={handleMethodChange}
          />

          {/* 输入内容区域 */}
          <View style={styles.inputContent}>
            {renderInputContent()}
          </View>
        </Card>

        {/* 图片预览 */}
        {selectedImage && (
          <ImagePreview
            image={selectedImage}
            onRemove={handleRemoveImage}
            onNext={handleNext}
          />
        )}

        {/* 快速体验横幅 */}
        {!selectedImage && (
          <View style={styles.quickStartContainer}>
            <QuickStartBanner
              onQuickStart={handleQuickStart}
              loading={loading}
            />
          </View>
        )}
      </ScrollView>
    </MainLayout>
  );
};

const styles = StyleSheet.create({
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
    paddingBottom: theme.spacing.xxxl,
  },
  header: {
    marginBottom: theme.spacing.lg,
  },
  titleContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  titleIcon: {
    width: 40,
    height: 40,
    borderRadius: 12,
    backgroundColor: `${theme.colors.primary}15`,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: theme.spacing.md,
  },
  titleIconInner: {
    width: 20,
    height: 20,
    borderRadius: 4,
    backgroundColor: theme.colors.primary,
  },
  titleRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  titleDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: theme.colors.primary,
    marginRight: theme.spacing.sm,
  },
  titleTextContainer: {
    height: 20,
    justifyContent: 'center',
  },
  titleLine: {
    width: 80,
    height: 4,
    borderRadius: 2,
    backgroundColor: theme.colors.text.primary,
  },
  inputContent: {
    minHeight: 200,
  },
  quickStartContainer: {
    marginTop: theme.spacing.lg,
  },
});

export default ImageInputScreen;
