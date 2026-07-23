import React, { useRef } from 'react';
import {
  View,
  ScrollView,
  TouchableOpacity,
  Text,
  StyleSheet,
  Animated,
} from 'react-native';
import { AnnotationFilter } from '../../types/dataset';
import { useResponsive } from '@/hooks/useResponsive';
import { theme } from '@/theme';

interface TypeFilterProps {
  selectedType: AnnotationFilter;
  onTypeChange: (type: AnnotationFilter) => void;
  /** 各类型数量（可选，来自统计信息） */
  counts?: Partial<Record<AnnotationFilter, number>>;
}

interface FilterButtonProps {
  label: string;
  count?: number;
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
        <Text
          style={[
            styles.filterText,
            compact && styles.filterTextCompact,
            isActive && styles.activeText,
          ]}
        >
          {label}
          {typeof count === 'number' && (
            <Text
              style={[
                styles.countText,
                compact && styles.countTextCompact,
                isActive && styles.activeCountText,
              ]}
            >
              {' '}
              {count}
            </Text>
          )}
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

  const filterTypes: { key: AnnotationFilter; label: string }[] = [
    { key: 'annotated', label: '已标注' },
    { key: 'unannotated', label: '未标注' },
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
        {filterTypes.map(type => {
          const count = counts?.[type.key];
          const isActive = selectedType === type.key;
          return (
            <FilterButton
              key={type.key}
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
    backgroundColor: theme.colors.background.primary,
    borderWidth: 2,
    borderColor: theme.colors.border.light,
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
    backgroundColor: theme.colors.secondary,
    borderColor: theme.colors.secondary,
    shadowColor: theme.colors.secondary,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 4,
  },
  filterText: {
    fontSize: 15,
    fontWeight: '600',
    color: theme.colors.text.secondary,
  },
  filterTextCompact: {
    fontSize: 14,
  },
  activeText: {
    color: theme.colors.text.inverse,
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
