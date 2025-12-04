import React from 'react';
import {
  View,
  Text,
  StyleSheet,
} from 'react-native';
import Card from '@/components/Card';
import { Dataset } from '../../types/dataset';

interface DatasetInfoCardProps {
  dataset: Dataset;
}

const DatasetInfoCard: React.FC<DatasetInfoCardProps> = ({
  dataset,
}) => {
  return (
    <Card padding={20} margin={0} borderRadius={12}>
      <View style={styles.gradientContainer}>
        {/* Title and Description */}
        <Text style={styles.name}>{dataset.name}</Text>
        <Text style={styles.description}>
          {dataset.description || '暂无描述'}
        </Text>

        {/* Stats Grid */}
        <View style={styles.statsGrid}>
          <View style={styles.statBox}>
            <Text style={styles.statValue}>{dataset.total_images}</Text>
            <Text style={styles.statLabel}>总计</Text>
          </View>
          <View style={styles.statBox}>
            <Text style={styles.statValue}>{dataset.foggy_count}</Text>
            <Text style={styles.statLabel}>有雾</Text>
          </View>
          <View style={styles.statBox}>
            <Text style={styles.statValue}>{dataset.clear_count}</Text>
            <Text style={styles.statLabel}>无雾</Text>
          </View>
          <View style={styles.statBox}>
            <Text style={styles.statValue}>{dataset.annotated_count}</Text>
            <Text style={styles.statLabel}>标注</Text>
          </View>
        </View>
      </View>
    </Card>
  );
};

const styles = StyleSheet.create({
  gradientContainer: {
    backgroundColor: '#14b8a6',
    borderRadius: 12,
    padding: 20,
    margin: -20,
  },
  name: {
    fontSize: 20,
    fontWeight: '700',
    color: '#ffffff',
    marginBottom: 8,
  },
  description: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.9)',
    lineHeight: 20,
    marginBottom: 20,
  },
  statsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 12,
  },
  statBox: {
    flex: 1,
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 8,
    paddingVertical: 12,
    paddingHorizontal: 8,
  },
  statValue: {
    fontSize: 20,
    fontWeight: '700',
    color: '#ffffff',
    marginBottom: 4,
  },
  statLabel: {
    fontSize: 12,
    color: 'rgba(255, 255, 255, 0.8)',
  },
});

export default DatasetInfoCard;