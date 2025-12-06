import React from 'react'
import { View, Text, Button, Image } from '@tarojs/components'
import Taro from '@tarojs/taro'
import { Arrow, Success } from '@taroify/icons'

import './AlgorithmSection.less'

const AlgorithmSection: React.FC = () => {
  const algorithmImageUrl = 'https://zhiyan-ai-agent-with-1258344702.cos.ap-guangzhou.tencentcos.cn/with/f49e4b9e-6079-4a0b-8f91-bcab5deec2c7/image_1763727581_1_3.jpg'

  const handleLearnMoreClick = () => {
    try {
      Taro.navigateTo({ url: '/pages/algorithm/index' })
    } catch (error) {
      console.warn('导航到算法页面不存在，将在实现后可用')
      Taro.showToast({
        title: '功能开发中',
        icon: 'none'
      })
    }
  }

  const algorithmFeatures = [
    {
      text: '智能推荐最适合的去雾算法'
    },
    {
      text: '实时对比不同算法的处理效果'
    },
    {
      text: '毫秒级处理速度，即时查看结果'
    },
    {
      text: '支持批量处理和参数自定义'
    }
  ]

  return (
    <View className='algorithm-section'>
      <View className='algorithm-content'>
        <View className='algorithm-text'>
          <Text className='section-title'>多算法智能选择</Text>
          <Text className='section-subtitle'>
            支持DCP、AOD-Net、DehazeNet等多种先进算法
          </Text>
          <View className='algorithm-features'>
            {algorithmFeatures.map((feature, index) => (
              <View key={index} className='feature-item'>
                <Success size='18' color='#34d399' />
                <Text className='feature-text'>{feature.text}</Text>
              </View>
            ))}
          </View>
          <Button
            className='learn-more-btn'
            onClick={handleLearnMoreClick}
          >
            了解更多算法详情
            <Arrow className='btn-icon' size='14' color='#3b82f6' />
          </Button>
        </View>
        <View className='algorithm-visual'>
          <Image
            src={algorithmImageUrl}
            className='algorithm-image'
            mode='widthFix'
            lazyLoad
          />
        </View>
      </View>
    </View>
  )
}

export default AlgorithmSection
