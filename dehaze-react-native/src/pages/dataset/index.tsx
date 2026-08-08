import React, { useState, useCallback } from 'react';
import { View, StyleSheet } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { AppHeader } from '@/layout';
import DatasetListSection from './components/DatasetListSection';
import DatasetDetailSection from './components/DatasetDetailSection';
import type { DatasetTreeNode } from './types/dataset';

const DatasetScreen: React.FC = () => {
  const navigation = useNavigation();
  const [currentView, setCurrentView] = useState<'list' | 'detail'>('list');
  const [selectedDatasetId, setSelectedDatasetId] = useState<number | null>(null);
  const [searchValue, setSearchValue] = useState('');

  const handleDatasetPress = useCallback((dataset: DatasetTreeNode) => {
    setSelectedDatasetId(dataset.id);
    setCurrentView('detail');
  }, []);

  const handleBack = useCallback(() => {
    setCurrentView('list');
    setSelectedDatasetId(null);
    setSearchValue('');
  }, []);

  const handleSearchChange = useCallback((text: string) => {
    setSearchValue(text);
  }, []);

  return (
    <View style={styles.container}>
      <AppHeader title="数据集管理" showBack onBackPress={() => navigation.goBack()} />
      <View style={styles.content}>
        {currentView === 'list' ? (
          <DatasetListSection
            onDatasetPress={handleDatasetPress}
            searchValue={searchValue}
            onSearchChange={handleSearchChange}
          />
        ) : selectedDatasetId ? (
          <DatasetDetailSection
            datasetId={selectedDatasetId}
            onBack={handleBack}
          />
        ) : null}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  content: {
    flex: 1,
  },
});

export default DatasetScreen;
