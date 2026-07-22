import React, { useState, useCallback, useRef } from "react";
import { View, Input } from "@tarojs/components";
import { Search } from "@taroify/icons";
import "./SearchBar.less";

interface SearchBarProps {
  placeholder?: string;
  value?: string;
  onSearch?: (value: string) => void;
  onClear?: () => void;
  showClear?: boolean;
  className?: string;
}

const SearchBar: React.FC<SearchBarProps> = ({
  placeholder = "搜索数据集或图片...",
  value = "",
  onSearch,
  onClear,
  showClear = true,
  className = "",
}) => {
  const [searchValue, setSearchValue] = useState(value);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // 防抖搜索
  const debouncedSearch = useCallback(
    (newValue: string) => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
      timerRef.current = setTimeout(() => {
        onSearch?.(newValue.trim());
      }, 500);
    },
    [onSearch]
  );

  const handleInput = (e: any) => {
    const newValue = e.detail.value;
    setSearchValue(newValue);
    debouncedSearch(newValue);
  };

  const handleConfirm = () => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
    }
    onSearch?.(searchValue.trim());
  };

  const handleClear = () => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
    }
    setSearchValue("");
    onClear?.();
  };

  return (
    <View className={`search-bar ${className}`}>
      <View className="search-input-wrapper">
        <Search className="search-icon" size="20" color="#9ca3af" />
        <Input
          className="search-input"
          placeholder={placeholder}
          value={searchValue}
          onInput={handleInput}
          onConfirm={handleConfirm}
        />
        {showClear && searchValue && (
          <View className="clear-btn" onClick={handleClear}>
            ×
          </View>
        )}
      </View>
    </View>
  );
};

export default SearchBar;
