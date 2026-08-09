import React from 'react';
import { View, StyleSheet } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { ToolsStackParamList } from '@/routes/types';
import { AppHeader } from '@/layout';
import DatasetDetailSection from './components/DatasetDetailSection';

type Props = NativeStackScreenProps<ToolsStackParamList, 'DatasetDetail'>;

const DatasetDetailScreen: React.FC<Props> = ({ navigation, route }) => {
  const { datasetId } = route.params;

  return (
    <View style={styles.container}>
      <AppHeader title="数据集详情" showBack onBackPress={() => navigation.goBack()} />
      <View style={styles.content}>
        <DatasetDetailSection datasetId={datasetId} onBack={() => navigation.goBack()} />
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

export default DatasetDetailScreen;