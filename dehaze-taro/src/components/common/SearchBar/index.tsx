import React, { useState, useCallback } from 'react'
import { View, Input } from '@tarojs/components'
import { Search } from '@taroify/icons'
import './SearchBar.less'

interface SearchBarProps {
  placeholder?: string
  value?: string
  onSearch?: (value: string) => void
  onClear?: () => void
  showClear?: boolean
  className?: string
}

const SearchBar: React.FC<SearchBarProps> = ({
  placeholder = '搜索数据集或图片...',
  value = '',
  onSearch,
  onClear,
  showClear = true,
  className = '',
}) => {
  const [searchValue, setSearchValue] = useState(value)
  const [isComposing, setIsComposing] = useState(false)

  // 防抖搜索
  const debouncedSearch = useCallback(
    (newValue: string) => {
      const timer = setTimeout(() => {
        onSearch?.(newValue.trim())
      }, 500)

      return () => clearTimeout(timer)
    },
    [onSearch]
  )

  const handleInput = (e: any) => {
    const newValue = e.detail.value
    setSearchValue(newValue)

    if (!isComposing) {
      debouncedSearch(newValue)
    }
  }

  const handleConfirm = () => {
    onSearch?.(searchValue.trim())
  }

  const handleClear = () => {
    setSearchValue('')
    onClear?.()
  }

  const handleCompositionStart = () => {
    setIsComposing(true)
  }

  const handleCompositionEnd = (e: any) => {
    setIsComposing(false)
    const newValue = e.detail.value
    setSearchValue(newValue)
    onSearch?.(newValue.trim())
  }

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
          onCompositionStart={handleCompositionStart}
          onCompositionEnd={handleCompositionEnd}
        />
        {showClear && searchValue && (
          <View className="clear-btn" onClick={handleClear}>
            ×
          </View>
        )}
      </View>
    </View>
  )
}

export default SearchBar