import React from 'react'
import { View, Text } from '@tarojs/components'
import { Fire, PhoneOutlined, BulbOutlined, ChartTrendingOutlined } from '@taroify/icons'

import './TechSpecs.less'

interface SpecCardProps {
  icon: React.ReactNode
  title: string
  value: string
  description: string
}

const SpecCard: React.FC<SpecCardProps> = ({ icon, title, value, description }) => {
  return (
    <View className='spec-card'>
      <View className='spec-icon'>
        {icon}
      </View>
      <Text className='spec-title'>{title}</Text>
      <Text className='spec-value'>{value}</Text>
      <View className='spec-desc'>
        <Text>{description}</Text>
      </View>
    </View>
  )
}

const TechSpecs: React.FC = () => {
  const specs = [
    {
      icon: <Fire size='28' color='#ffffff' />,
      title: '高性能',
      value: '60fps',
      description: '流畅运行，响应时间<200ms'
    },
    {
      icon: <PhoneOutlined size='28' color='#ffffff' />,
      title: '全平台',
      value: '100%',
      description: '完美适配手机、平板、桌面'
    },
    {
      icon: <BulbOutlined size='28' color='#ffffff' />,
      title: '智能算法',
      value: '8+',
      description: '支持多种先进去雾算法'
    },
    {
      icon: <ChartTrendingOutlined size='28' color='#ffffff' />,
      title: '专业评估',
      value: '5+',
      description: '多维度定量分析指标'
    }
  ]

  return (
    <View className='tech-specs-section'>
      <View className='specs-grid'>
        {specs.map((spec, index) => (
          <SpecCard
            key={index}
            icon={spec.icon}
            title={spec.title}
            value={spec.value}
            description={spec.description}
          />
        ))}
      </View>
    </View>
  )
}

export default TechSpecs