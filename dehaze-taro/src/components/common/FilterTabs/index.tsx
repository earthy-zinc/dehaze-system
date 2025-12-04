import React from 'react'
import { View, Text } from '@tarojs/components'
import './FilterTabs.less'

export interface FilterTab {
  key: string
  label: string
  count?: number
  active?: boolean
}

interface FilterTabsProps {
  tabs: FilterTab[]
  activeKey: string
  onChange: (key: string) => void
  className?: string
}

const FilterTabs: React.FC<FilterTabsProps> = ({
  tabs,
  activeKey,
  onChange,
  className = '',
}) => {
  const handleTabClick = (key: string) => {
    if (key !== activeKey) {
      onChange(key)
    }
  }

  return (
    <View className={`filter-tabs ${className}`}>
      <View className="tabs-container">
        {tabs.map((tab) => (
          <View
            key={tab.key}
            className={`tab-item ${activeKey === tab.key ? 'active' : ''}`}
            onClick={() => handleTabClick(tab.key)}
          >
            <Text className="tab-text">{tab.label}</Text>
            {tab.count !== undefined && (
              <Text className="tab-count">{tab.count}</Text>
            )}
          </View>
        ))}
      </View>
    </View>
  )
}

export default FilterTabs