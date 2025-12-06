/**
 * 历史记录组件
 */

import React from 'react'
import { View, Text, Image } from '@tarojs/components'
import { SwipeCell, Button, Loading } from '@taroify/core'
import { Arrow, DeleteOutlined } from '@taroify/icons'
import { HistoryRecord, GroupedHistory } from '../../services/types'
import { groupHistoryByDate, formatTimestamp } from '../../services/history'
import EmptyState from '@/components/common/EmptyState'
import './HistoryList.less'

interface HistoryListProps {
  records: HistoryRecord[]
  loading?: boolean
  onSelect: (record: HistoryRecord) => void
  onDelete: (id: number) => void
  onClear: () => void
}

const HistoryList: React.FC<HistoryListProps> = ({
  records,
  loading = false,
  onSelect,
  onDelete,
  onClear,
}) => {
  // 按时间分组
  const groupedRecords = groupHistoryByDate(records)

  if (loading) {
    return (
      <View className='history-list'>
        <View className='loading-container'>
          <Loading size='32px' />
          <Text className='loading-text'>加载中...</Text>
        </View>
      </View>
    )
  }

  if (records.length === 0) {
    return (
      <View className='history-list'>
        <EmptyState
          type='history'
          title='暂无历史记录'
          description='处理过的图片会显示在这里'
        />
      </View>
    )
  }

  return (
    <View className='history-list'>
      {/* 头部操作栏 */}
      <View className='history-header'>
        <Text className='header-title'>最近处理的图片</Text>
        <View className='clear-btn' onClick={onClear}>
          <DeleteOutlined size='14' />
          <Text>清空</Text>
        </View>
      </View>

      {/* 分组列表 */}
      {groupedRecords.map((group) => (
        <View key={group.title} className='history-group'>
          <Text className='group-title'>{group.title}</Text>
          <View className='group-list'>
            {group.records.map((record) => (
              <SwipeCell key={record.id}>
                <View
                  className='history-item'
                  onClick={() => onSelect(record)}
                >
                  <View className='item-thumbnail'>
                    <Image
                      className='thumbnail-image'
                      src={record.originalImage}
                      mode='aspectFill'
                    />
                    {record.status === 'success' && record.resultImage && (
                      <View className='result-badge'>
                        <Text>已处理</Text>
                      </View>
                    )}
                    {record.status === 'failed' && (
                      <View className='result-badge failed'>
                        <Text>失败</Text>
                      </View>
                    )}
                  </View>
                  <View className='item-info'>
                    <Text className='item-name'>{record.fileName || '未命名图片'}</Text>
                    <Text className='item-time'>{formatTimestamp(record.timestamp)}</Text>
                    {record.algorithm && (
                      <Text className='item-algorithm'>{record.algorithm}</Text>
                    )}
                  </View>
                  <View className='item-arrow'>
                    <Arrow size='16' color='#9ca3af' />
                  </View>
                </View>
                <SwipeCell.Actions side='right'>
                  <Button
                    variant='contained'
                    color='danger'
                    onClick={() => onDelete(record.id)}
                  >
                    删除
                  </Button>
                </SwipeCell.Actions>
              </SwipeCell>
            ))}
          </View>
        </View>
      ))}

      {/* 底部提示 */}
      <View className='history-footer'>
        <Text className='footer-text'>最多保存最近 20 条记录</Text>
      </View>
    </View>
  )
}

export default HistoryList
