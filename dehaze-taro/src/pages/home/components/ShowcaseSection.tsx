import React from 'react'
import { View, Text } from '@tarojs/components'

import ComparisonItem from './ComparisonItem'
import './ShowcaseSection.less'

const ShowcaseSection: React.FC = () => {
  const showcaseImageUrl = 'https://zhiyan-ai-agent-with-1258344702.cos.ap-guangzhou.tencentcos.cn/with/20b8704f-d37e-45b9-a6c8-3c5d297e8a98/image_1763727568_3_3.jpg'

  return (
    <View className='showcase-section'>
      <View className='showcase-header'>
        <Text className='section-title'>一键去雾，效果显著</Text>
        <Text className='section-subtitle'>
          智能算法自动识别雾霾程度，精准还原图像细节
        </Text>
      </View>
      <View className='comparison-showcase'>
        <ComparisonItem imageUrl={showcaseImageUrl} />
      </View>
    </View>
  )
}

export default ShowcaseSection