import React, { useRef } from 'react';
import {
  View,
  ScrollView,
  TouchableOpacity,
  Text,
  StyleSheet,
  Animated,
} from 'react-native';
import { ImageTypeFilter } from '../../types/dataset';
import { useResponsive } from '@/hooks/useResponsive';

interface TypeFilterProps {
  selectedType: ImageTypeFilter;
  onTypeChange: (type: ImageTypeFilter) => void;
  counts: {
    all: number;
    foggy: number;
    clear: number;
    annotated: number;
  };
}

interface FilterButtonProps {
  type: ImageTypeFilter;
  label: string;
  count: number;
  isActive: boolean;
  onPress: () => void;
  compact?: boolean;
}

const FilterButton: React.FC<FilterButtonProps> = ({
  label,
  count,
  isActive,
  onPress,
  compact,
}) => {
  const scaleAnim = useRef(new Animated.Value(1)).current;

  const handlePressIn = () => {
    Animated.spring(scaleAnim, {
      toValue: 0.95,
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

  return (
    <Animated.View style={{ transform: [{ scale: scaleAnim }] }}>
      <TouchableOpacity
        style={[
          styles.filterButton,
          compact && styles.filterButtonCompact,
          isActive && styles.activeButton,
        ]}
        onPress={onPress}
        onPressIn={handlePressIn}
        onPressOut={handlePressOut}
        activeOpacity={1}
      >
        <Text style={[
          styles.filterText,
          compact && styles.filterTextCompact,
          isActive && styles.activeText,
        ]}>
          {label}
          <Text style={[
            styles.countText,
            compact && styles.countTextCompact,
            isActive && styles.activeCountText,
          ]}>
            {' '}{count}
          </Text>
        </Text>
      </TouchableOpacity>
    </Animated.View>
  );
};

const TypeFilter: React.FC<TypeFilterProps> = ({
  selectedType,
  onTypeChange,
  counts,
}) => {
  const { isMobile, containerPadding, spacing } = useResponsive();

  const filterTypes = [
    { key: 'all' as ImageTypeFilter, label: '全部' },
    { key: 'foggy' as ImageTypeFilter, label: '有雾' },
    { key: 'clear' as ImageTypeFilter, label: '无雾' },
    { key: 'annotated' as ImageTypeFilter, label: '标注' },
  ];

  return (
    <View style={styles.container}>
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={[
          styles.scrollContent,
          { paddingHorizontal: containerPadding, gap: spacing },
        ]}
      >
        {filterTypes.map((type) => {
          const count = counts[type.key];
          const isActive = selectedType === type.key;

          return (
            <FilterButton
              key={type.key}
              type={type.key}
              label={type.label}
              count={count}
              isActive={isActive}
              onPress={() => onTypeChange(type.key)}
              compact={isMobile}
            />
          );
        })}
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    marginVertical: 16,
  },
  scrollContent: {
    alignItems: 'center',
  },
  filterButton: {
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 24,
    backgroundColor: '#ffffff',
    borderWidth: 2,
    borderColor: '#e5e7eb',
    alignItems: 'center',
    justifyContent: 'center',
    minWidth: 90,
  },
  filterButtonCompact: {
    paddingHorizontal: 16,
    paddingVertical: 10,
    minWidth: 72,
  },
  activeButton: {
    backgroundColor: '#14b8a6',
    borderColor: '#14b8a6',
    shadowColor: '#14b8a6',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 4,
  },
  filterText: {
    fontSize: 15,
    fontWeight: '600',
    color: '#6b7280',
  },
  filterTextCompact: {
    fontSize: 14,
  },
  activeText: {
    color: '#ffffff',
  },
  countText: {
    fontSize: 13,
    opacity: 0.8,
  },
  countTextCompact: {
    fontSize: 12,
  },
  activeCountText: {
    opacity: 0.9,
  },
});

export default TypeFilter;