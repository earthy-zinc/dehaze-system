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
  ImageTypeFilter,
} from '../../types/dataset';
import { datasetApi } from '../../services/datasetApi';

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
  const [selectedType, setSelectedType] = useState<ImageTypeFilter>('all');
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

  const handleTypeChange = useCallback((type: ImageTypeFilter) => {
    setSelectedType(type);
  }, []);

  const handleSearchChange = useCallback((text: string) => {
    setSearchValue(text);
    setCurrentPage(1);
  }, []);

  /** 将数据项按图片类型过滤并扁平化为图片列表 */
  const flatImages: DatasetImage[] = React.useMemo(() => {
    const result: DatasetImage[] = [];
    for (const item of items) {
      if (selectedType === 'all' || selectedType === 'clear') {
        if (item.clearImage) result.push(item.clearImage);
      }
      if (selectedType === 'all' || selectedType === 'hazy') {
        if (item.hazyImages) {
          for (const hazy of item.hazyImages) result.push(hazy);
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
              await datasetApi.createExportTask(dataset.id, {
                structure: 'by_item',
                includeTypes: ['clear', 'hazy'],
                includeThumbnail: false,
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

  if (error && !dataset && !isLoading) {
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
        <EmptyState icon="search-plus" title="加载失败" description={error} />
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
        {dataset && (
          <Button
            title="导出"
            onPress={handleExport}
            variant="primary"
            icon={<Icon name="export" size={14} color="#ffffff" />}
          />
        )}
      </View>

      {/* Dataset Info */}
      {dataset && (
        <View style={styles.infoSection}>
          <DatasetInfoCard dataset={dataset} />
        </View>
      )}

      {/* Type Filter */}
      {dataset && (
        <TypeFilter
          selectedType={selectedType}
          onTypeChange={handleTypeChange}
          counts={
            dataset.statistics
              ? {
                  all: dataset.statistics.fileCount,
                  clear: dataset.statistics.clearCount,
                  hazy: dataset.statistics.hazyCount,
                }
              : undefined
          }
        />
      )}

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

      {/* Loading State */}
      {isLoading && (
        <View style={styles.loadingContainer}>
          <LoadingSpinner size="large" text="加载数据中..." />
        </View>
      )}

      {/* Empty State */}
      {!isLoading && !isLoadingImages && flatImages.length === 0 && dataset && (
        <View style={styles.emptyContainer}>
          <EmptyState
            icon="image"
            title="暂无图片"
            description={
              searchValue
                ? '未找到匹配的数据项'
                : selectedType !== 'all'
                ? `暂无${selectedType === 'clear' ? '清晰' : '有雾'}图片`
                : '该数据集还没有图片'
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
