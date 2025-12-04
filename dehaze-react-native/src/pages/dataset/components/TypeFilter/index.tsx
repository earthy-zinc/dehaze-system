import React from 'react';
import {
  View,
  ScrollView,
  TouchableOpacity,
  Text,
  StyleSheet,
} from 'react-native';
import { ImageTypeFilter } from '../../types/dataset';

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

const TypeFilter: React.FC<TypeFilterProps> = ({
  selectedType,
  onTypeChange,
  counts,
}) => {
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
        contentContainerStyle={styles.scrollContent}
      >
        {filterTypes.map((type) => {
          const count = counts[type.key];
          const isActive = selectedType === type.key;

          return (
            <TouchableOpacity
              key={type.key}
              style={[
                styles.filterButton,
                isActive && styles.activeButton,
              ]}
              onPress={() => onTypeChange(type.key)}
              activeOpacity={0.8}
            >
              <Text style={[
                styles.filterText,
                isActive && styles.activeText,
              ]}>
                {type.label}
                <Text style={[
                  styles.countText,
                  isActive && styles.activeCountText,
                ]}>
                  {' '}{count}
                </Text>
              </Text>
            </TouchableOpacity>
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
    paddingHorizontal: 20,
    gap: 12,
  },
  filterButton: {
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 20,
    backgroundColor: '#ffffff',
    borderWidth: 2,
    borderColor: '#e5e7eb',
    alignItems: 'center',
    justifyContent: 'center',
    minWidth: 80,
  },
  activeButton: {
    backgroundColor: '#14b8a6',
    borderColor: '#14b8a6',
    shadowColor: '#14b8a6',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 8,
    elevation: 4,
  },
  filterText: {
    fontSize: 14,
    fontWeight: '500',
    color: '#6b7280',
  },
  activeText: {
    color: '#ffffff',
  },
  countText: {
    fontSize: 12,
    opacity: 0.8,
  },
  activeCountText: {
    opacity: 0.9,
  },
});

export default TypeFilter;