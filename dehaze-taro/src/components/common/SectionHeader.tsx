import React from 'react'
import { View, Text } from '@tarojs/components'

import './SectionHeader.less'

interface SectionHeaderProps {
  title: string
  subtitle?: string
  align?: 'left' | 'center' | 'right'
  className?: string
}

const SectionHeader: React.FC<SectionHeaderProps> = ({
  title,
  subtitle,
  align = 'left',
  className = ''
}) => {
  const containerClass = [
    'section-header',
    `section-header--${align}`,
    className
  ].filter(Boolean).join(' ')

  return (
    <View className={containerClass}>
      <Text className='section-header__title'>{title}</Text>
      {subtitle && (
        <Text className='section-header__subtitle'>{subtitle}</Text>
      )}
    </View>
  )
}

export default SectionHeader