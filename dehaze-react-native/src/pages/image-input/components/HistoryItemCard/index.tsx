/**
 * 历史记录卡片组件
 */

import React, { useRef } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Animated,
} from 'react-native';
import ImageLoader from '@/components/ImageLoader';
import Icon from '@/components/Icon';
import { theme } from '@/theme';
import { HistoryRecord } from '../../types/imageInput';
import { historyStorage } from '../../services/historyStorage';
import { extractFilename } from '@/utils/url';

interface HistoryItemCardProps {
  record: HistoryRecord;
  onPress: (record: HistoryRecord) => void;
  onDelete: (id: number) => void;
}

const HistoryItemCard: React.FC<HistoryItemCardProps> = ({
  record,
  onPress,
  onDelete,
}) => {
  const scaleAnim = useRef(new Animated.Value(1)).current;

  const handlePressIn = () => {
    Animated.spring(scaleAnim, {
      toValue: 0.98,
      useNativeDriver: true,
      tension: 100,
      friction: 8,
    }).start();
  };

  const handlePressOut = () => {
    Animated.spring(scaleAnim, {
      toValue: 1,
      useNativeDriver: true,
      tension: 100,
      friction: 8,
    }).start();
  };

  const thumbnailUrl = record.originalThumbnailUrl || '';
  const filename = extractFilename(record.originalImageUrl);
  const formattedTime = historyStorage.formatTimestamp(record.createTime);
  const isSuccess = !!record.resultImageUrl;

  return (
    <TouchableOpacity
      onPress={() => onPress(record)}
      onPressIn={handlePressIn}
      onPressOut={handlePressOut}
      activeOpacity={1}
    >
      <Animated.View
        style={[
          styles.container,
          { transform: [{ scale: scaleAnim }] },
        ]}
      >
        {/* 缩略图 */}
        <View style={styles.thumbnailContainer}>
          <ImageLoader
            source={{ uri: thumbnailUrl }}
            style={styles.thumbnail}
            resizeMode="cover"
          />

          {/* 状态指示器 */}
          <View
            style={[
              styles.statusIndicator,
              isSuccess ? styles.statusSuccess : styles.statusFailed,
            ]}
          >
            <Icon
              name={isSuccess ? 'checkmark' : 'close'}
              size={10}
              color="#fff"
            />
          </View>
        </View>

        {/* 信息区域 */}
        <View style={styles.infoContainer}>
          <Text style={styles.filename} numberOfLines={1}>
            {filename}
          </Text>

          <Text style={styles.time}>{formattedTime}</Text>

          {record.algorithmName && (
            <View style={styles.algorithmTag}>
              <Icon name="code" size={12} color={theme.colors.primary} />
              <Text style={styles.algorithmText}>{record.algorithmName}</Text>
            </View>
          )}
        </View>

        {/* 操作按钮 */}
        <View style={styles.actions}>
          <TouchableOpacity
            onPress={() => onPress(record)}
            style={styles.actionButton}
            hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
          >
            <Icon name="refresh" size={18} color={theme.colors.primary} />
          </TouchableOpacity>

          <TouchableOpacity
            onPress={() => onDelete(record.id)}
            style={styles.actionButton}
            hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
          >
            <Icon name="trash" size={18} color={theme.colors.status.error} />
          </TouchableOpacity>
        </View>
      </Animated.View>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: theme.colors.background.primary,
    borderRadius: theme.layout.borderRadius.lg,
    padding: theme.spacing.sm,
    marginBottom: theme.spacing.sm,
    ...theme.layout.shadows.sm,
  },
  thumbnailContainer: {
    width: 64,
    height: 64,
    borderRadius: theme.layout.borderRadius.md,
    overflow: 'hidden',
    position: 'relative',
  },
  thumbnail: {
    width: '100%',
    height: '100%',
  },
  statusIndicator: {
    position: 'absolute',
    bottom: 4,
    right: 4,
    width: 16,
    height: 16,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
  },
  statusSuccess: {
    backgroundColor: theme.colors.status.success,
  },
  statusFailed: {
    backgroundColor: theme.colors.status.error,
  },
  infoContainer: {
    flex: 1,
    marginLeft: theme.spacing.md,
    justifyContent: 'center',
  },
  filename: {
    fontSize: theme.typography.sizes.body,
    fontWeight: theme.typography.weights.medium,
    color: theme.colors.text.primary,
    marginBottom: 2,
  },
  time: {
    fontSize: theme.typography.sizes.caption,
    color: theme.colors.text.tertiary,
    marginBottom: 4,
  },
  algorithmTag: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  algorithmText: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.primary,
  },
  actions: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.sm,
  },
  actionButton: {
    padding: theme.spacing.xs,
  },
});

export default HistoryItemCard;
