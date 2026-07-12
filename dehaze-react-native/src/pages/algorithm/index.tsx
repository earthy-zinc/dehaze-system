import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '@/routes/types';
import { MainLayout } from '@/layout';

type Props = NativeStackScreenProps<RootStackParamList, 'Algorithm'>;

const AlgorithmScreen: React.FC<Props> = () => {
  return (
    <MainLayout title="算法详情">
      <View style={styles.content}>
        <Text style={styles.title}>算法详情</Text>
        <Text style={styles.description}>此页面正在开发中...</Text>
      </View>
    </MainLayout>
  );
};

const styles = StyleSheet.create({
  content: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
  },
  description: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
  },
});

export default AlgorithmScreen;