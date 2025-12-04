import React from 'react'
import { View, Text } from '@tarojs/components'
import Taro from '@tarojs/taro'

import './WorkflowSection.less'

interface WorkflowStepProps {
  number: string
  icon: React.ReactNode
  title: string
  description: string
  target: string
}

const WorkflowStep: React.FC<WorkflowStepProps> = ({
  number,
  icon,
  title,
  description,
  target
}) => {
  const handleClick = () => {
    try {
      Taro.navigateTo({ url: `/pages/${target}/index` })
    } catch (error) {
      console.warn(`导航到 ${target} 页面不存在，将在实现后可用`)
      Taro.showToast({
        title: '功能开发中',
        icon: 'none'
      })
    }
  }

  return (
    <View className='workflow-step' onClick={handleClick}>
      <View className='step-number'>{number}</View>
      <View className='step-icon'>{icon}</View>
      <Text className='step-title'>{title}</Text>
      <View className='step-desc'>
        {description.split('\n').map((line, index) => (
          <Text key={index}>{line}</Text>
        ))}
      </View>
    </View>
  )
}

const WorkflowSection: React.FC = () => {
  return (
    <View className='workflow-section'>
      <View className='features-header'>
        <Text className='section-title'>强大的功能生态</Text>
        <Text className='section-subtitle'>从输入到输出，每一步都精心设计</Text>
      </View>

      <View className='workflow-container'>
        <WorkflowStep
          number='01'
          icon={<Text className='step-icon-text'>🖼️</Text>}
          title='图像输入'
          description='支持上传、拍照、样例图片\n多种输入方式随心选择'
          target='image-input'
        />

        <View className='workflow-arrow'>
          <Text>→</Text>
        </View>

        <WorkflowStep
          number='02'
          icon={<Text className='step-icon-text'>🧠</Text>}
          title='智能算法'
          description='多种去雾算法可选\nAI智能推荐最优方案'
          target='algorithm-select'
        />

        <View className='workflow-arrow'>
          <Text>→</Text>
        </View>

        <WorkflowStep
          number='03'
          icon={<Text className='step-icon-text'>⚡</Text>}
          title='一键处理'
          description='毫秒级处理速度\n实时预览处理效果'
          target='processing'
        />
      </View>
    </View>
  )
}

export default WorkflowSection