import React, { useState, useCallback } from 'react';
import { View, StyleSheet } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import type { ToolsStackParamList } from '@/routes/types';
import { AppHeader } from '@/layout';
import DatasetListSection from './components/DatasetListSection';
import type { DatasetTreeNode } from './types/dataset';

const DatasetScreen: React.FC = () => {
  const navigation = useNavigation<NativeStackNavigationProp<ToolsStackParamList, 'Dataset'>>();
  const [searchValue, setSearchValue] = useState('');

  const handleDatasetPress = useCallback(
    (dataset: DatasetTreeNode) => {
      navigation.navigate('DatasetDetail', { datasetId: dataset.id });
    },
    [navigation],
  );

  const handleSearchChange = useCallback((text: string) => {
    setSearchValue(text);
  }, []);

  return (
    <View style={styles.container}>
      <AppHeader title="数据集管理" showBack onBackPress={() => navigation.goBack()} />
      <View style={styles.content}>
        <DatasetListSection
          onDatasetPress={handleDatasetPress}
          searchValue={searchValue}
          onSearchChange={handleSearchChange}
        />
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
