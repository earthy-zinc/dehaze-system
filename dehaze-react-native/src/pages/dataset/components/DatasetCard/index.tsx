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
import Card from '@/components/Card';
import { useResponsive } from '@/hooks/useResponsive';
import { Dataset } from '../../types/dataset';

interface DatasetCardProps {
  dataset: Dataset;
  onPress: (dataset: Dataset) => void;
}

const DatasetCard: React.FC<DatasetCardProps> = ({
  dataset,
  onPress,
}) => {
  const { isMobile } = useResponsive();
  const scaleAnim = useRef(new Animated.Value(1)).current;

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));

    if (days === 0) return '今天';
    if (days === 1) return '昨天';
    if (days < 7) return `${days}天前`;

    return date.toLocaleDateString('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
    });
  };

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

  // 响应式缩略图尺寸
  const thumbnailSize = isMobile ? 100 : 120;

  return (
    <TouchableOpacity
      onPress={() => onPress(dataset)}
      onPressIn={handlePressIn}
      onPressOut={handlePressOut}
      activeOpacity={1}
      style={styles.container}
    >
      <Animated.View style={{ transform: [{ scale: scaleAnim }] }}>
        <Card padding={0} margin={0} borderRadius={12}>
          <View style={styles.cardContent}>
            {/* Thumbnail */}
            <View style={[
              styles.thumbnailContainer,
              { width: thumbnailSize, height: thumbnailSize },
            ]}>
              <ImageLoader
                source={{ uri: dataset.thumbnail }}
                style={styles.thumbnail}
                resizeMode="cover"
              />
            </View>

            {/* Content */}
            <View style={styles.content}>
              <Text style={styles.title} numberOfLines={1}>
                {dataset.name}
              </Text>

              <Text style={styles.description} numberOfLines={2}>
                {dataset.description || '暂无描述'}
              </Text>

              <View style={styles.stats}>
                <View style={styles.statItem}>
                  <Icon name="image" size={14} color="#14b8a6" />
                  <Text style={styles.statText}>{dataset.total_images}</Text>
                </View>

                <View style={styles.statItem}>
                  <Icon name="clock" size={14} color="#9ca3af" />
                  <Text style={styles.statText}>{formatDate(dataset.created_at)}</Text>
                </View>
              </View>
            </View>

            {/* Arrow indicator */}
            <View style={styles.arrowContainer}>
              <Icon name="chevron-right" size={16} color="#9ca3af" />
            </View>
          </View>
        </Card>
      </Animated.View>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
    marginBottom: 12,
  },
  cardContent: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  thumbnailContainer: {
    borderTopLeftRadius: 12,
    borderBottomLeftRadius: 12,
    overflow: 'hidden',
  },
  thumbnail: {
    width: '100%',
    height: '100%',
    backgroundColor: '#f3f4f6',
  },
  content: {
    flex: 1,
    padding: 16,
    justifyContent: 'space-between',
  },
  title: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 6,
  },
  description: {
    fontSize: 14,
    color: '#6b7280',
    lineHeight: 20,
    marginBottom: 12,
    flex: 1,
  },
  stats: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
  },
  statItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  statText: {
    fontSize: 13,
    color: '#6b7280',
  },
  arrowContainer: {
    paddingRight: 12,
  },
});

export default DatasetCard;