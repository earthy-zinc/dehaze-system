/**
 * 图标兼容性组件
 * 当taroify图标不可用时，使用文本或emoji替代
 */

import React from 'react'
import { Text } from '@tarojs/components'
import { ArrowOutlined } from '@taroify/icons'

interface IconWrapperProps {
  icon?: string
  size?: number | string
  color?: string
  fallback?: string
  className?: string
}

const IconWrapper: React.FC<IconWrapperProps> = ({
  icon,
  size = 24,
  color = 'inherit',
  fallback = '📦',
  className = ''
}) => {
  // 尝试使用taroify图标，如果失败则使用fallback
  try {
    return icon ? (
      <Text
        className={className}
        style={{
          fontSize: size,
          color,
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}
      >
        {icon}
      </Text>
    ) : (
      <Text
        className={className}
        style={{
          fontSize: size,
          color,
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}
      >
        {fallback}
      </Text>
    )
  } catch (error) {
    // 如果图标渲染失败，使用fallback
    return (
      <Text
        className={className}
        style={{
          fontSize: size,
          color,
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}
      >
        {fallback}
      </Text>
    )
  }
}

export default IconWrapper