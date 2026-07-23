/**
 * 历史记录列表组件
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  SectionList,
  Alert,
} from 'react-native';
import Icon from '@/components/Icon';
import EmptyState from '@/components/EmptyState';
import { theme } from '@/theme';
import type { SelectedImage } from '@/types/image';
import { HistoryRecord, HistoryGroup } from '../../types/imageInput';
import { historyStorage } from '../../services/historyStorage';
import HistoryItemCard from '../HistoryItemCard';

interface HistoryListProps {
  onSelectRecord: (image: SelectedImage) => void;
}

const HistoryList: React.FC<HistoryListProps> = ({
  onSelectRecord,
}) => {
  const [historyGroups, setHistoryGroups] = useState<HistoryGroup[]>([]);
  const [loading, setLoading] = useState(true);

  // 加载历史记录
  const loadHistory = useCallback(async () => {
    setLoading(true);
    try {
      const history = await historyStorage.getHistory();
      const groups = historyStorage.groupHistoryByDate(history);
      setHistoryGroups(groups);
    } catch {
      setHistoryGroups([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  // 处理记录选择
  const handleRecordPress = useCallback((record: HistoryRecord) => {
    const selectedImage: SelectedImage = {
      id: record.id.toString(),
      url: record.originalImageUrl || '',
      thumbUrl: record.originalThumbnailUrl,
      name: record.algorithmName || '历史图片',
      source: 'history',
      algorithmId: record.algorithmId,
      algorithmParams: record.algorithmParams,
    };

    onSelectRecord(selectedImage);
  }, [onSelectRecord]);

  // 处理删除记录
  const handleDeleteRecord = useCallback((id: number) => {
    Alert.alert(
      '删除记录',
      '确定要删除这条历史记录吗？',
      [
        { text: '取消', style: 'cancel' },
        {
          text: '删除',
          style: 'destructive',
          onPress: async () => {
            try {
              await historyStorage.deleteRecord(id);
              loadHistory();
            } catch (error) {
              Alert.alert('错误', '删除失败，请重试');
            }
          },
        },
      ]
    );
  }, [loadHistory]);

  // 处理清空历史
  const handleClearHistory = useCallback(() => {
    Alert.alert(
      '清空历史记录',
      '确定要清空所有历史记录吗？此操作不可恢复。',
      [
        { text: '取消', style: 'cancel' },
        {
          text: '清空',
          style: 'destructive',
          onPress: async () => {
            try {
              await historyStorage.clearHistory();
              setHistoryGroups([]);
            } catch (error) {
              Alert.alert('错误', '清空失败，请重试');
            }
          },
        },
      ]
    );
  }, []);

  // 渲染分组标题
  const renderSectionHeader = ({ section }: { section: HistoryGroup }) => (
    <View style={styles.sectionHeader}>
      <Text style={styles.sectionTitle}>{section.title}</Text>
    </View>
  );

  // 渲染记录卡片
  const renderItem = ({ item }: { item: HistoryRecord }) => (
    <HistoryItemCard
      record={item}
      onPress={handleRecordPress}
      onDelete={handleDeleteRecord}
    />
  );

  // 渲染空状态
  const renderEmpty = () => (
    <EmptyState
      icon="time"
      title="暂无历史记录"
      description="处理过的图片会显示在这里"
    />
  );

  // 计算总记录数
  const totalRecords = historyGroups.reduce(
    (sum, group) => sum + group.data.length,
    0
  );

  return (
    <View style={styles.container}>
      {/* 头部 */}
      <View style={styles.header}>
        <View style={styles.headerLeft}>
          <Icon name="time" size={18} color={theme.colors.text.secondary} />
          <Text style={styles.headerText}>
            最近处理的图片 ({totalRecords})
          </Text>
        </View>

        {totalRecords > 0 && (
          <TouchableOpacity
            onPress={handleClearHistory}
            style={styles.clearButton}
            hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
          >
            <Icon name="trash" size={16} color={theme.colors.status.error} />
            <Text style={styles.clearButtonText}>清空</Text>
          </TouchableOpacity>
        )}
      </View>

      {/* 列表 */}
      {historyGroups.length > 0 ? (
        <SectionList
          sections={historyGroups}
          renderItem={renderItem}
          renderSectionHeader={renderSectionHeader}
          keyExtractor={item => item.id.toString()}
          showsVerticalScrollIndicator={false}
          stickySectionHeadersEnabled={false}
          scrollEnabled={false} // 禁用内部滚动
        />
      ) : (
        !loading && renderEmpty()
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: theme.spacing.md,
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.sm,
  },
  headerText: {
    fontSize: theme.typography.sizes.body,
    color: theme.colors.text.secondary,
  },
  clearButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  clearButtonText: {
    fontSize: theme.typography.sizes.body,
    color: theme.colors.status.error,
  },
  sectionHeader: {
    paddingVertical: theme.spacing.sm,
  },
  sectionTitle: {
    fontSize: theme.typography.sizes.caption,
    fontWeight: theme.typography.weights.semibold,
    color: theme.colors.text.tertiary,
    textTransform: 'uppercase',
  },
});

export default HistoryList;
