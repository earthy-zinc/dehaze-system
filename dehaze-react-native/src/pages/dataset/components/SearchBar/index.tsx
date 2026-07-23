import React, { useState, useCallback } from 'react';
import {
  View,
  TextInput,
  TouchableOpacity,
  StyleSheet,
} from 'react-native';
import Icon from '@/components/Icon';

interface SearchBarProps {
  value: string;
  onChangeText: (text: string) => void;
  placeholder?: string;
  onClear?: () => void;
}

const SearchBar: React.FC<SearchBarProps> = ({
  value,
  onChangeText,
  placeholder = '搜索数据集或图片...',
  onClear,
}) => {
  const [showClear, setShowClear] = useState(false);

  const handleTextChange = useCallback((text: string) => {
    onChangeText(text);
    setShowClear(text.length > 0);
  }, [onChangeText]);

  const handleClear = useCallback(() => {
    onChangeText('');
    setShowClear(false);
    onClear?.();
  }, [onChangeText, onClear]);

  const handleFocus = useCallback(() => {
    setShowClear(value.length > 0);
  }, [value]);

  return (
    <View style={styles.container}>
      <Icon name="search-plus" size={20} color="#9ca3af" style={styles.searchIcon} />
      <TextInput
        style={styles.input}
        value={value}
        onChangeText={handleTextChange}
        onFocus={handleFocus}
        placeholder={placeholder}
        placeholderTextColor="#9ca3af"
        selectionColor="#14b8a6"
      />
      {showClear && (
        <TouchableOpacity
          style={styles.clearButton}
          onPress={handleClear}
          activeOpacity={0.8}
        >
          <Icon name="times" size={16} color="#9ca3af" />
        </TouchableOpacity>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    position: 'relative',
    backgroundColor: '#ffffff',
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#e5e7eb',
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  searchIcon: {
    position: 'absolute',
    left: 16,
    top: '50%',
    transform: [{ translateY: -10 }],
    zIndex: 1,
  },
  input: {
    paddingLeft: 40,
    paddingRight: 40,
    fontSize: 16,
    color: '#1f2937',
    height: 44,
  },
  clearButton: {
    position: 'absolute',
    right: 16,
    top: '50%',
    transform: [{ translateY: -12 }],
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: '#f3f4f6',
    alignItems: 'center',
    justifyContent: 'center',
  },
});

export default SearchBar;