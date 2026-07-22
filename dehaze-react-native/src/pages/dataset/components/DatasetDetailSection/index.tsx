import React, { useState, useCallback, useEffect, useRef } from 'react';
import {
  View,
  StyleSheet,
  Alert,
} from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import Button from '@/components/Button';
import Icon from '@/components/Icon';
import DatasetInfoCard from '../DatasetInfoCard';
import TypeFilter from '../TypeFilter';
import ImageGrid from '../ImageGrid';
import ImageViewer from '../ImageViewer';
import SearchBar from '../SearchBar';
import LoadingSpinner from '@/components/LoadingSpinner';
import EmptyState from '@/components/EmptyState';
import type {
  Dataset,
  DatasetItem,
  DatasetImage,
  AnnotationFilter,
} from '../../types/dataset';
import { isImageAnnotated } from '../../types/dataset';
import { datasetApi } from '../../services/datasetApi';
import { taskApi } from '../../../task/services/taskApi';

interface DatasetDetailSectionProps {
  datasetId: number;
  onBack: () => void;
  /** 导出任务创建回调 */
  onExport?: (dataset: Dataset) => void;
}

const PAGE_SIZE = 20;

const DatasetDetailSection: React.FC<DatasetDetailSectionProps> = ({
  datasetId,
  onBack,
  onExport,
}) => {
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [items, setItems] = useState<DatasetItem[]>([]);
  const [selectedType, setSelectedType] = useState<AnnotationFilter>('annotated');
  const [searchValue, setSearchValue] = useState('');
  const [isLoading, setLoading] = useState(true);
  const [isLoadingImages, setIsLoadingImages] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [currentPage, setCurrentPage] = useState(1);
  const [selectedImage, setSelectedImage] = useState<DatasetImage | null>(null);
  const [selectedItem, setSelectedItem] = useState<DatasetItem | null>(null);
  const [error, setError] = useState<string | null>(null);

  // 搜索防抖
  const searchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const loadDatasetDetail = useCallback(async () => {
    try {
      const detail = await datasetApi.fetchDatasetDetail(datasetId);
      setDataset(detail);
    } catch (err: any) {
      setError(err?.msg || err?.message || '加载数据集详情失败');
    }
  }, [datasetId]);

  const loadItems = useCallback(
    async (page = 1, isRefresh = false) => {
      if (isLoadingImages && !isRefresh) return;
      try {
        setIsLoadingImages(true);
        if (isRefresh) setRefreshing(true);

        const result = await datasetApi.fetchDatasetItems({
          datasetId,
          keyword: searchValue.trim() || undefined,
          pageNum: page,
          pageSize: PAGE_SIZE,
        });

        const list = result.list || [];
        if (page === 1) {
          setItems(list);
        } else {
          setItems(prev => [...prev, ...list]);
        }
        setHasMore(list.length >= PAGE_SIZE);
        setCurrentPage(page);
      } catch (err: any) {
        setError(err?.msg || err?.message || '加载数据项失败');
      } finally {
        setIsLoadingImages(false);
        setRefreshing(false);
      }
    },
    [datasetId, searchValue, isLoadingImages],
  );

  const handleTypeChange = useCallback((type: AnnotationFilter) => {
    setSelectedType(type);
  }, []);

  const handleSearchChange = useCallback((text: string) => {
    setSearchValue(text);
    setCurrentPage(1);
  }, []);

  /** 将数据项按标注状态过滤并扁平化为图片列表 */
  const flatImages: DatasetImage[] = React.useMemo(() => {
    const result: DatasetImage[] = [];
    for (const item of items) {
      if (item.clearImage) {
        const img = item.clearImage;
        if (selectedType === 'annotated' && isImageAnnotated(img.hazeLevel)) {
          result.push(img);
        } else if (selectedType === 'unannotated' && !isImageAnnotated(img.hazeLevel)) {
          result.push(img);
        }
      }
      if (item.hazyImages) {
        for (const hazy of item.hazyImages) {
          if (selectedType === 'annotated' && isImageAnnotated(hazy.hazeLevel)) {
            result.push(hazy);
          } else if (selectedType === 'unannotated' && !isImageAnnotated(hazy.hazeLevel)) {
            result.push(hazy);
          }
        }
      }
    }
    return result;
  }, [items, selectedType]);

  /** 根据图片 id 回溯所属数据项 */
  const findItemByImageId = useCallback(
    (imageId: number): DatasetItem | null => {
      for (const item of items) {
        if (item.clearImage?.id === imageId) return item;
        if (item.hazyImages?.some(h => h.id === imageId)) return item;
      }
      return null;
    },
    [items],
  );

  const handleImagePress = useCallback(
    (image: DatasetImage) => {
      setSelectedImage(image);
      setSelectedItem(findItemByImageId(image.id));
    },
    [findItemByImageId],
  );

  const handleImageClose = useCallback(() => {
    setSelectedImage(null);
    setSelectedItem(null);
  }, []);

  const handleLoadMore = useCallback(() => {
    if (hasMore && !isLoadingImages) {
      loadItems(currentPage + 1);
    }
  }, [hasMore, isLoadingImages, currentPage, loadItems]);

  const handleRefresh = useCallback(() => {
    Promise.all([loadDatasetDetail(), loadItems(1, true)]);
  }, [loadDatasetDetail, loadItems]);

  const handleExport = useCallback(() => {
    if (!dataset) return;
    if (onExport) {
      onExport(dataset);
    } else {
      Alert.alert('提示', `将创建数据集"${dataset.name}"的导出任务，导出完成后可在任务中心下载。`, [
        { text: '取消', style: 'cancel' },
        {
          text: '确认导出',
          onPress: async () => {
            try {
              await taskApi.create({
                type: 'dataset_export',
                targetId: dataset.id,
                options: {
                  structure: 'by_item',
                  includeTypes: ['clear', 'hazy'],
                  includeThumbnail: false,
                },
              });
              Alert.alert('已创建', '导出任务已创建，请到任务中心查看进度');
            } catch (err: any) {
              Alert.alert(
                '导出失败',
                err?.msg || err?.message || '请稍后重试',
              );
            }
          },
        },
      ]);
    }
  }, [dataset, onExport]);

  // 类型切换时无需重新请求（前端过滤）
  useEffect(() => {
    if (searchTimerRef.current) clearTimeout(searchTimerRef.current);
    searchTimerRef.current = setTimeout(() => {
      loadItems(1);
    }, 350);
    return () => {
      if (searchTimerRef.current) clearTimeout(searchTimerRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchValue]);

  useFocusEffect(
    useCallback(() => {
      setLoading(true);
      Promise.all([loadDatasetDetail(), loadItems(1)]).finally(() =>
        setLoading(false),
      );
    }, [loadDatasetDetail, loadItems]),
  );

  if (isLoading) {
    return (
      <View style={styles.container}>
        <View style={styles.header}>
          <Button
            title="返回"
            onPress={onBack}
            variant="secondary"
            icon={<Icon name="back" size={14} color="#3b82f6" />}
          />
        </View>
        <View style={styles.loadingContainer}>
          <LoadingSpinner size="large" text="加载数据中..." />
        </View>
      </View>
    );
  }

  if (error || !dataset) {
    return (
      <View style={styles.container}>
        <View style={styles.header}>
          <Button
            title="返回"
            onPress={onBack}
            variant="secondary"
            icon={<Icon name="back" size={14} color="#3b82f6" />}
          />
        </View>
        <EmptyState
          icon="search-plus"
          title={error ? '加载失败' : '数据集不存在'}
          description={error || '该数据集可能已被删除或无法访问'}
        />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Button
          title="返回"
          onPress={onBack}
          variant="secondary"
          icon={<Icon name="back" size={14} color="#3b82f6" />}
        />
        <Button
          title="导出"
          onPress={handleExport}
          variant="primary"
          icon={<Icon name="export" size={14} color="#ffffff" />}
        />
      </View>

      {/* Dataset Info */}
      <View style={styles.infoSection}>
        <DatasetInfoCard dataset={dataset} />
      </View>

      {/* Type Filter */}
      <TypeFilter
        selectedType={selectedType}
        onTypeChange={handleTypeChange}
        counts={
          dataset.statistics
            ? {
                annotated: dataset.statistics.annotatedCount,
                unannotated: dataset.statistics.unannotatedCount,
              }
            : undefined
        }
      />

      {/* Search Bar */}
      <View style={styles.searchContainer}>
        <SearchBar
          value={searchValue}
          onChangeText={handleSearchChange}
          placeholder="搜索数据项..."
        />
      </View>

      {/* Image Grid */}
      <ImageGrid
        images={flatImages}
        onImagePress={handleImagePress}
        onEndReached={handleLoadMore}
        onRefresh={handleRefresh}
        refreshing={refreshing}
        isLoading={isLoadingImages}
      />

      {/* Empty State */}
      {!isLoadingImages && flatImages.length === 0 && (
        <View style={styles.emptyContainer}>
          <EmptyState
            icon="image"
            title="暂无图片"
            description={
              searchValue
                ? '未找到匹配的数据项'
                : `暂无${selectedType === 'annotated' ? '已标注' : '未标注'}图片`
            }
          />
        </View>
      )}

      {/* Image Viewer */}
      <ImageViewer
        visible={!!selectedImage}
        onClose={handleImageClose}
        image={selectedImage}
        item={selectedItem}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f9fafb',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    paddingBottom: 8,
    backgroundColor: '#ffffff',
    borderBottomWidth: 1,
    borderBottomColor: '#f3f4f6',
  },
  infoSection: {
    marginHorizontal: 20,
    marginVertical: 16,
  },
  searchContainer: {
    marginHorizontal: 20,
    marginBottom: 16,
  },
  loadingContainer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(249, 250, 251, 0.8)',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 10,
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 40,
  },
});

export default DatasetDetailSection;
