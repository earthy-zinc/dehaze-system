import React, { useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Animated,
  Easing,
} from 'react-native';
import Card from '@/components/Card';
import { Dataset } from '../../types/dataset';
import { useResponsive } from '@/hooks/useResponsive';

interface DatasetInfoCardProps {
  dataset: Dataset;
}

const DatasetInfoCard: React.FC<DatasetInfoCardProps> = ({
  dataset,
}) => {
  const { isMobile, fontScale } = useResponsive();

  // 入场动画
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

  // 响应式字体大小
  const titleFontSize = isMobile ? 18 : 20 * fontScale;

  return (
    <Animated.View style={{
      opacity: fadeAnim,
      transform: [{ translateY: slideAnim }],
    }}>
      <Card padding={20} margin={0} borderRadius={16}>
        <View style={styles.gradientContainer}>
          {/* Title and Description */}
          <Text style={[styles.name, { fontSize: titleFontSize }]}>
            {dataset.name}
          </Text>
          <Text style={styles.description}>
            {dataset.description || '暂无描述'}
          </Text>

          {/* Stats Grid */}
          <View style={[
            styles.statsGrid,
            isMobile && styles.statsGridCompact,
          ]}>
            <View style={[styles.statBox, isMobile && styles.statBoxCompact]}>
              <Text style={[styles.statValue, isMobile && styles.statValueCompact]}>
                {dataset.total_images}
              </Text>
              <Text style={styles.statLabel}>总计</Text>
            </View>
            <View style={[styles.statBox, isMobile && styles.statBoxCompact]}>
              <Text style={[styles.statValue, isMobile && styles.statValueCompact]}>
                {dataset.foggy_count}
              </Text>
              <Text style={styles.statLabel}>有雾</Text>
            </View>
            <View style={[styles.statBox, isMobile && styles.statBoxCompact]}>
              <Text style={[styles.statValue, isMobile && styles.statValueCompact]}>
                {dataset.clear_count}
              </Text>
              <Text style={styles.statLabel}>无雾</Text>
            </View>
            <View style={[styles.statBox, isMobile && styles.statBoxCompact]}>
              <Text style={[styles.statValue, isMobile && styles.statValueCompact]}>
                {dataset.annotated_count}
              </Text>
              <Text style={styles.statLabel}>标注</Text>
            </View>
          </View>
        </View>
      </Card>
    </Animated.View>
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
    color: '#ffffff',
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