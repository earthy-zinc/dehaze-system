import React, { useState, useCallback } from 'react';
import {
  View,
  FlatList,
  StyleSheet,
  RefreshControl,
} from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import DatasetCard from '../DatasetCard';
import SearchBar from '../SearchBar';
import LoadingSpinner from '@/components/LoadingSpinner';
import EmptyState from '@/components/EmptyState';
import { Dataset } from '../../types/dataset';
import { datasetApi } from '../../services/datasetApi';

interface DatasetListSectionProps {
  onDatasetPress: (dataset: Dataset) => void;
  searchValue: string;
  onSearchChange: (text: string) => void;
}

const DatasetListSection: React.FC<DatasetListSectionProps> = ({
  onDatasetPress,
  searchValue,
  onSearchChange,
}) => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [isLoading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadDatasets = useCallback(async (isRefresh = false) => {
    try {
      if (isRefresh) {
        setRefreshing(true);
      } else if (!isRefresh && datasets.length === 0) {
        setLoading(true);
      }
      setError(null);

      const response = await datasetApi.fetchDatasets(1, searchValue);

      if (response.code === 0) {
        setDatasets(response.data.list);
      } else {
        setError('加载数据集失败');
      }
    } catch (err) {
      setError('网络错误，请重试');
      console.error('加载数据集失败:', err);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [searchValue, datasets.length]);

  useFocusEffect(
    useCallback(() => {
      loadDatasets();
    }, [loadDatasets])
  );

  const handleRefresh = useCallback(() => {
    loadDatasets(true);
  }, [loadDatasets]);

  const renderItem = useCallback(({ item }: { item: Dataset }) => (
    <DatasetCard
      dataset={item}
      onPress={onDatasetPress}
    />
  ), [onDatasetPress]);

  const keyExtractor = useCallback((item: Dataset) => item.id.toString(), []);

  const renderEmpty = useCallback(() => {
    if (isLoading) return null;

    return (
      <EmptyState
        icon="database"
        title="暂无数据集"
        description={searchValue ? '未找到匹配的数据集' : '还没有添加任何数据集'}
      />
    );
  }, [isLoading, searchValue]);

  if (error && datasets.length === 0) {
    return (
      <View style={styles.container}>
        <SearchBar
          value={searchValue}
          onChangeText={onSearchChange}
        />
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
      {/* Search Bar */}
      <View style={styles.searchContainer}>
        <SearchBar
          value={searchValue}
          onChangeText={onSearchChange}
        />
      </View>

      {/* Dataset List */}
      <FlatList
        data={datasets}
        renderItem={renderItem}
        keyExtractor={keyExtractor}
        contentContainerStyle={styles.listContainer}
        showsVerticalScrollIndicator={false}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={handleRefresh}
            tintColor="#14b8a6"
            colors={['#14b8a6']}
          />
        }
        ListEmptyComponent={renderEmpty}
        ListFooterComponent={
          isLoading ? (
            <View style={styles.loadingContainer}>
              <LoadingSpinner size="large" color="#14b8a6" />
            </View>
          ) : null
        }
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f9fafb',
  },
  searchContainer: {
    backgroundColor: '#ffffff',
    paddingHorizontal: 20,
    paddingTop: 16,
    paddingBottom: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#f3f4f6',
  },
  listContainer: {
    padding: 20,
    paddingTop: 12,
  },
  loadingContainer: {
    paddingVertical: 20,
  },
});

export default DatasetListSection;