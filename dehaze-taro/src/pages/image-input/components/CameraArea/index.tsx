/**
 * 拍照区域组件
 */

import React from 'react'
import { View, Text } from '@tarojs/components'
import { CameraOutlined } from '@taroify/icons'
import { Loading } from '@taroify/core'
import './CameraArea.less'

interface CameraAreaProps {
  onCapture: () => void
  loading?: boolean
}

const CameraArea: React.FC<CameraAreaProps> = ({ onCapture, loading = false }) => {
  const handleClick = () => {
    if (!loading) {
      onCapture()
    }
  }

  return (
    <View className='camera-area'>
      <View className='camera-zone'>
        <View className='camera-icon'>
          <CameraOutlined size='64' color='#9ca3af' />
        </View>

        <Text className='camera-desc'>点击下方按钮打开相机</Text>
        <Text className='camera-hint'>拍摄需要去雾处理的图片</Text>

        <View
          className={`camera-btn ${loading ? 'loading' : ''}`}
          onClick={handleClick}
        >
          {loading ? (
            <>
              <Loading size='18px' />
              <Text>拍照中...</Text>
            </>
          ) : (
            <>
              <CameraOutlined size='20' />
              <Text>打开相机</Text>
            </>
          )}
        </View>
      </View>

      <View className='camera-tips'>
        <View className='tip-title'>拍摄建议</View>
        <View className='tip-list'>
          <View className='tip-item'>
            <Text className='tip-dot'>•</Text>
            <Text>选择雾霾天气或有雾的场景</Text>
          </View>
          <View className='tip-item'>
            <Text className='tip-dot'>•</Text>
            <Text>保持手机稳定，避免抖动</Text>
          </View>
          <View className='tip-item'>
            <Text className='tip-dot'>•</Text>
            <Text>确保光线充足，提高去雾效果</Text>
          </View>
        </View>
      </View>
    </View>
  )
}

export default CameraArea
