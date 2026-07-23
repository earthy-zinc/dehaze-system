import React, { useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Animated,
  Easing,
} from 'react-native';
import Card from '@/components/Card';
import Badge from '@/components/Badge';
import { Dataset } from '../../types/dataset';
import { useResponsive } from '@/hooks/useResponsive';
import { theme } from '@/theme';

interface DatasetInfoCardProps {
  dataset: Dataset;
}

const DatasetInfoCard: React.FC<DatasetInfoCardProps> = ({ dataset }) => {
  const { isMobile, fontScale } = useResponsive();

  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(20)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 500,
        easing: Easing.out(Easing.cubic),
        useNativeDriver: true,
      }),
      Animated.timing(slideAnim, {
        toValue: 0,
        duration: 500,
        easing: Easing.out(Easing.cubic),
        useNativeDriver: true,
      }),
    ]).start();
  }, [fadeAnim, slideAnim]);

  const titleFontSize = isMobile ? 18 : 20 * fontScale;
  const stats = dataset.statistics;
  const itemCount = stats?.itemCount ?? dataset.total ?? 0;
  const fileCount = stats?.fileCount ?? 0;
  const annotatedCount = stats?.annotatedCount ?? 0;
  const unannotatedCount = stats?.unannotatedCount ?? 0;

  return (
    <Animated.View
      style={{ opacity: fadeAnim, transform: [{ translateY: slideAnim }] }}
    >
      <Card padding={20} margin={0} borderRadius={16}>
        <View style={styles.gradientContainer}>
          {/* 类型 + 状态 */}
          <View style={styles.tagsRow}>
            {!!dataset.type && (
              <Badge text={dataset.type} variant="info" size="small" />
            )}
            <Badge
              text={dataset.status === 0 ? '禁用' : '启用'}
              variant={dataset.status === 0 ? 'secondary' : 'success'}
              size="small"
            />
          </View>

          {/* 名称 + 描述 */}
          <Text style={[styles.name, { fontSize: titleFontSize }]}>
            {dataset.name}
          </Text>
          <Text style={styles.description}>
            {dataset.description || '暂无描述'}
          </Text>

          {/* 统计信息 */}
          <View style={[styles.statsGrid, isMobile && styles.statsGridCompact]}>
            <View style={[styles.statBox, isMobile && styles.statBoxCompact]}>
              <Text style={[styles.statValue, isMobile && styles.statValueCompact]}>
                {itemCount}
              </Text>
              <Text style={styles.statLabel}>数据项</Text>
            </View>
            <View style={[styles.statBox, isMobile && styles.statBoxCompact]}>
              <Text style={[styles.statValue, isMobile && styles.statValueCompact]}>
                {fileCount}
              </Text>
              <Text style={styles.statLabel}>文件</Text>
            </View>
            <View style={[styles.statBox, isMobile && styles.statBoxCompact]}>
              <Text style={[styles.statValue, isMobile && styles.statValueCompact]}>
                {annotatedCount}
              </Text>
              <Text style={styles.statLabel}>已标注</Text>
            </View>
            <View style={[styles.statBox, isMobile && styles.statBoxCompact]}>
              <Text style={[styles.statValue, isMobile && styles.statValueCompact]}>
                {unannotatedCount}
              </Text>
              <Text style={styles.statLabel}>未标注</Text>
            </View>
          </View>
        </View>
      </Card>
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  gradientContainer: {
    backgroundColor: theme.colors.secondary,
    borderRadius: 12,
    padding: 20,
    margin: -20,
  },
  tagsRow: {
    flexDirection: 'row',
    gap: 8,
    marginBottom: 12,
  },
  name: {
    fontWeight: '700',
    color: theme.colors.text.inverse,
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
  statsGridCompact: {
    gap: 8,
  },
  statBox: {
    flex: 1,
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    borderRadius: 10,
    paddingVertical: 14,
    paddingHorizontal: 8,
  },
  statBoxCompact: {
    paddingVertical: 10,
    paddingHorizontal: 6,
  },
  statValue: {
    fontSize: 22,
    fontWeight: '700',
    color: theme.colors.text.inverse,
    marginBottom: 4,
  },
  statValueCompact: {
    fontSize: 18,
  },
  statLabel: {
    fontSize: 12,
    color: 'rgba(255, 255, 255, 0.85)',
    fontWeight: '500',
  },
});

export default DatasetInfoCard;
