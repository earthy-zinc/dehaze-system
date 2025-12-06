import React, { useState, useCallback } from 'react';
import {
  View,
  StyleSheet,
} from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { RootStackParamList } from '@/routes/navigator';
import { MainLayout } from '@/layout';
import DatasetListSection from './components/DatasetListSection';
import DatasetDetailSection from './components/DatasetDetailSection';
import { Dataset } from './types/dataset';

type Props = NativeStackScreenProps<RootStackParamList, 'Dataset'>;

const DatasetScreen: React.FC<Props> = () => {
  const [currentView, setCurrentView] = useState<'list' | 'detail'>('list');
  const [selectedDatasetId, setSelectedDatasetId] = useState<number | null>(null);
  const [searchValue, setSearchValue] = useState('');

  const handleDatasetPress = useCallback((dataset: Dataset) => {
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
    <MainLayout title="数据集管理">
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
    </MainLayout>
  );
};

const styles = StyleSheet.create({
  content: {
    flex: 1,
  },
});

export default DatasetScreen;