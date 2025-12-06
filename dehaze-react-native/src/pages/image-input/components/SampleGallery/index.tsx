/**
 * 样例图片库组件
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  FlatList,
  StyleSheet,
  ActivityIndicator,
  Text,
} from 'react-native';
import { useResponsive } from '@/hooks/useResponsive';
import { theme } from '@/theme';
import { SampleImage, SampleCategory, SelectedImage } from '../../types/imageInput';
import { imageInputApi } from '../../services/imageInputApi';
import SampleCategoryTabs from '../SampleCategoryTabs';
import SampleImageCard from '../SampleImageCard';

interface SampleGalleryProps {
  onSelectSample: (image: SelectedImage) => void;
}

const SampleGallery: React.FC<SampleGalleryProps> = ({
  onSelectSample,
}) => {
  const { columns } = useResponsive();
  const [category, setCategory] = useState<SampleCategory>('all');
  const [samples, setSamples] = useState<SampleImage[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingImage, setLoadingImage] = useState(false);

  // 加载样例图片
  const loadSamples = useCallback(async (cat: SampleCategory) => {
    setLoading(true);
    try {
      const response = await imageInputApi.fetchSamples(cat);
      if (response.code === 0) {
        setSamples(response.data.list);
      }
    } catch (error) {
      console.error('Failed to load samples:', error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadSamples(category);
  }, [category, loadSamples]);

  // 处理分类切换
  const handleCategoryChange = useCallback((newCategory: SampleCategory) => {
    setCategory(newCategory);
  }, []);

  // 处理样例选择
  const handleSamplePress = useCallback(async (sample: SampleImage) => {
    setLoadingImage(true);
    try {
      // 获取图片尺寸
      let width = 1920;
      let height = 1080;

      try {
        const size = await imageInputApi.getImageSize(sample.url);
        width = size.width;
        height = size.height;
      } catch (e) {
        // 使用默认值
      }

      const selectedImage: SelectedImage = {
        id: Date.now().toString(),
        uri: sample.url,
        filename: `${sample.name}.jpg`,
        width,
        height,
        fileSize: 0, // 远程图片无法获取文件大小
        source: 'sample',
        sampleInfo: sample,
      };

      onSelectSample(selectedImage);
    } catch (error) {
      console.error('Failed to select sample:', error);
    } finally {
      setLoadingImage(false);
    }
  }, [onSelectSample]);

  // 渲染样例卡片
  const renderSampleCard = useCallback(({ item }: { item: SampleImage }) => (
    <SampleImageCard
      sample={item}
      onPress={handleSamplePress}
    />
  ), [handleSamplePress]);

  // 渲染空状态
  const renderEmpty = () => (
    <View style={styles.emptyContainer}>
      <Text style={styles.emptyText}>暂无样例图片</Text>
    </View>
  );

  // 渲染加载状态
  const renderLoading = () => (
    <View style={styles.loadingContainer}>
      <ActivityIndicator size="large" color={theme.colors.primary} />
      <Text style={styles.loadingText}>加载中...</Text>
    </View>
  );

  return (
    <View style={styles.container}>
      {/* 分类标签 */}
      <SampleCategoryTabs
        currentCategory={category}
        onCategoryChange={handleCategoryChange}
      />

      {/* 图片网格 */}
      {loading ? (
        renderLoading()
      ) : (
        <FlatList
          data={samples}
          renderItem={renderSampleCard}
          keyExtractor={item => item.id.toString()}
          numColumns={columns}
          key={columns} // 强制在列数变化时重新渲染
          contentContainerStyle={styles.listContent}
          showsVerticalScrollIndicator={false}
          ListEmptyComponent={renderEmpty}
          scrollEnabled={false} // 禁用内部滚动，由外层 ScrollView 控制
        />
      )}

      {/* 加载图片遮罩 */}
      {loadingImage && (
        <View style={styles.loadingOverlay}>
          <View style={styles.loadingBox}>
            <ActivityIndicator size="large" color={theme.colors.primary} />
            <Text style={styles.loadingOverlayText}>加载样例图片...</Text>
          </View>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  listContent: {
    paddingHorizontal: -6,
  },
  emptyContainer: {
    padding: theme.spacing.xxxl,
    alignItems: 'center',
  },
  emptyText: {
    fontSize: theme.typography.sizes.body,
    color: theme.colors.text.tertiary,
  },
  loadingContainer: {
    padding: theme.spacing.xxxl,
    alignItems: 'center',
  },
  loadingText: {
    fontSize: theme.typography.sizes.body,
    color: theme.colors.text.secondary,
    marginTop: theme.spacing.md,
  },
  loadingOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 10,
  },
  loadingBox: {
    backgroundColor: theme.colors.background.primary,
    padding: theme.spacing.xxl,
    borderRadius: theme.layout.borderRadius.lg,
    alignItems: 'center',
    ...theme.layout.shadows.lg,
  },
  loadingOverlayText: {
    fontSize: theme.typography.sizes.body,
    color: theme.colors.text.secondary,
    marginTop: theme.spacing.md,
  },
});

export default SampleGallery;
