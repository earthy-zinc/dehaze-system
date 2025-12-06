/**
 * 样例图片库组件
 */

import React from 'react'
import { View, Text, Image } from '@tarojs/components'
import { Loading } from '@taroify/core'
import FilterTabs from '@/components/common/FilterTabs'
import { SampleImage, SampleCategory } from '../../services/types'
import { categoryTabs, difficultyColorMap } from '../../services/sampleData'
import './SampleGallery.less'

interface SampleGalleryProps {
  samples: SampleImage[]
  category: SampleCategory
  loading?: boolean
  onCategoryChange: (category: SampleCategory) => void
  onSelect: (sample: SampleImage) => void
}

const SampleGallery: React.FC<SampleGalleryProps> = ({
  samples,
  category,
  loading = false,
  onCategoryChange,
  onSelect,
}) => {
  // 转换为 FilterTabs 需要的格式
  const tabs = categoryTabs.map(tab => ({
    key: tab.key,
    label: tab.label,
  }))

  return (
    <View className='sample-gallery'>
      {/* 分类筛选 */}
      <View className='filter-section'>
        <FilterTabs
          tabs={tabs}
          activeKey={category}
          onChange={(key) => onCategoryChange(key as SampleCategory)}
        />
      </View>

      {/* 图片网格 */}
      {loading ? (
        <View className='loading-container'>
          <Loading size='32px' />
          <Text className='loading-text'>加载中...</Text>
        </View>
      ) : samples.length === 0 ? (
        <View className='empty-container'>
          <Text className='empty-text'>暂无样例图片</Text>
        </View>
      ) : (
        <View className='sample-grid'>
          {samples.map((sample) => (
            <View
              key={sample.id}
              className='sample-card'
              onClick={() => onSelect(sample)}
            >
              <View className='sample-image-wrapper'>
                <Image
                  className='sample-image'
                  src={sample.url}
                  mode='aspectFill'
                  lazyLoad
                />
                <View className={`difficulty-badge ${difficultyColorMap[sample.difficulty]}`}>
                  {sample.difficulty}
                </View>
              </View>
              <View className='sample-info'>
                <Text className='sample-name'>{sample.name}</Text>
                {sample.sceneType && (
                  <Text className='sample-scene'>{sample.sceneType}</Text>
                )}
              </View>
            </View>
          ))}
        </View>
      )}

      {/* 快速体验提示 */}
      <View className='quick-tip'>
        <Text className='tip-text'>点击任意图片即可快速体验去雾效果</Text>
      </View>
    </View>
  )
}

export default SampleGallery
