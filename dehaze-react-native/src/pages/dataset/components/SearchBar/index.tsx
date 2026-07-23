import React, { useState, useCallback } from 'react';
import {
  View,
  TextInput,
  TouchableOpacity,
  StyleSheet,
} from 'react-native';
import Icon from '@/components/Icon';
import { theme } from '@/theme';

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
      <Icon name="search-plus" size={20} color={theme.colors.text.tertiary} style={styles.searchIcon} />
      <TextInput
        style={styles.input}
        value={value}
        onChangeText={handleTextChange}
        onFocus={handleFocus}
        placeholder={placeholder}
        placeholderTextColor={theme.colors.text.tertiary}
        selectionColor={theme.colors.secondary}
      />
      {showClear && (
        <TouchableOpacity
          style={styles.clearButton}
          onPress={handleClear}
          activeOpacity={0.8}
        >
          <Icon name="times" size={16} color={theme.colors.text.tertiary} />
        </TouchableOpacity>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    position: 'relative',
    backgroundColor: theme.colors.background.primary,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: theme.colors.border.light,
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
    color: theme.colors.text.primary,
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
    backgroundColor: theme.colors.background.tertiary,
    alignItems: 'center',
    justifyContent: 'center',
  },
});

export default SearchBar;
