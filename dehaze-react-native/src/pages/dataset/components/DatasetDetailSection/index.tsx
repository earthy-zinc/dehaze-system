import React, { useState, useCallback, useEffect } from 'react';
import {
  View,
  StyleSheet,
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
import { Dataset, DatasetImage, ImageTypeFilter } from '../../types/dataset';
import { datasetApi } from '../../services/datasetApi';

interface DatasetDetailSectionProps {
  datasetId: number;
  onBack: () => void;
}

const DatasetDetailSection: React.FC<DatasetDetailSectionProps> = ({
  datasetId,
  onBack,
}) => {
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [images, setImages] = useState<DatasetImage[]>([]);
  const [selectedType, setSelectedType] = useState<ImageTypeFilter>('all');
  const [searchValue, setSearchValue] = useState('');
  const [isLoading, setLoading] = useState(true);
  const [isLoadingImages, setIsLoadingImages] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [currentPage, setCurrentPage] = useState(1);
  const [selectedImage, setSelectedImage] = useState<DatasetImage | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadDatasetDetail = useCallback(async () => {
    try {
      const response = await datasetApi.fetchDatasetDetail(datasetId);
      if (response.code === 0) {
        setDataset(response.data);
      } else {
        setError('加载数据集详情失败');
      }
    } catch (err) {
      setError('网络错误，请重试');
      console.error('加载数据集详情失败:', err);
    }
  }, [datasetId]);

  const loadImages = useCallback(async (page = 1, isRefresh = false) => {
    if (isLoadingImages && !isRefresh) return;

    try {
      setIsLoadingImages(true);
      if (isRefresh) {
        setRefreshing(true);
      }

      const response = await datasetApi.fetchDatasetImages(
        datasetId,
        page,
        selectedType,
        searchValue
      );

      if (response.code === 0) {
        if (page === 1) {
          setImages(response.data.list);
        } else {
          setImages(prev => [...prev, ...response.data.list]);
        }

        setHasMore(response.data.page < response.data.total_pages);
        setCurrentPage(response.data.page);
      } else {
        setError('加载图片失败');
      }
    } catch (err) {
      setError('网络错误，请重试');
      console.error('加载图片失败:', err);
    } finally {
      setIsLoadingImages(false);
      setRefreshing(false);
    }
  }, [datasetId, selectedType, searchValue, isLoadingImages]);

  const handleTypeChange = useCallback((type: ImageTypeFilter) => {
    setSelectedType(type);
    setCurrentPage(1);
    setImages([]);
  }, []);

  const handleSearchChange = useCallback((text: string) => {
    setSearchValue(text);
    setCurrentPage(1);
    setImages([]);
  }, []);

  const handleImagePress = useCallback((image: DatasetImage) => {
    setSelectedImage(image);
  }, []);

  const handleImageClose = useCallback(() => {
    setSelectedImage(null);
  }, []);

  const handleLoadMore = useCallback(() => {
    if (hasMore && !isLoadingImages) {
      loadImages(currentPage + 1);
    }
  }, [hasMore, isLoadingImages, currentPage, loadImages]);

  const handleRefresh = useCallback(() => {
    loadImages(1, true);
    loadDatasetDetail();
  }, [loadImages, loadDatasetDetail]);

  useEffect(() => {
    loadImages(1);
  }, [selectedType, searchValue, loadImages]);

  useFocusEffect(
    useCallback(() => {
      setLoading(true);
      loadDatasetDetail();
      loadImages(1);
      setLoading(false);
    }, [loadDatasetDetail, loadImages])
  );

  if (error && !dataset) {
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
          title="加载失败"
          description={error}
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
          counts={{
            all: dataset.total_images,
            foggy: dataset.foggy_count,
            clear: dataset.clear_count,
            annotated: dataset.annotated_count,
          }}
        />
      )}

      {/* Search Bar */}
      <View style={styles.searchContainer}>
        <SearchBar
          value={searchValue}
          onChangeText={handleSearchChange}
          placeholder="搜索图片..."
        />
      </View>

      {/* Image Grid */}
      <ImageGrid
        images={images}
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
      {!isLoading && !isLoadingImages && images.length === 0 && dataset && (
        <View style={styles.emptyContainer}>
          <EmptyState
            icon="image"
            title="暂无图片"
            description={
              searchValue
                ? '未找到匹配的图片'
                : selectedType !== 'all'
                ? `暂无${selectedType === 'foggy' ? '有雾' : selectedType === 'clear' ? '无雾' : '标注'}图片`
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