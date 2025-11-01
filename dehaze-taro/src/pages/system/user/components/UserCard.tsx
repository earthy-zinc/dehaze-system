import React from 'react';
import { View, Image, Text } from '@tarojs/components';
import { Tag, Button, Space } from '@taroify/core';
import { Edit, Delete, Lock, Phone } from '@taroify/icons';
import type { UserPageVO } from 'dehaze-sdk-js';
import { usePermission } from '@/hooks/usePermission';
import './UserCard.scss';

interface UserCardProps {
  user: UserPageVO;
  onEdit: () => void;
  onDelete: () => void;
  onResetPassword: () => void;
}

const UserCard: React.FC<UserCardProps> = ({
  user,
  onEdit,
  onDelete,
  onResetPassword,
}) => {
  const { hasPermission } = usePermission();

  const getGenderLabel = (genderLabel?: string) => {
    const genderMap: Record<string, string> = {
      '男': '男',
      '女': '女',
      '未知': '未知',
    };
    return genderMap[genderLabel || '未知'] || '未知';
  };

  const getGenderColor = (genderLabel?: string) => {
    const colorMap: Record<string, 'primary' | 'danger' | 'default'> = {
      '男': 'primary',
      '女': 'danger',
      '未知': 'default',
    };
    return colorMap[genderLabel || '未知'] || 'default';
  };

  return (
    <View className="user-card">
      <View className="user-card__header">
        <View className="user-avatar">
          <Image
            src={user.avatar || '/assets/default-avatar.png'}
            className="user-avatar__img"
            mode="aspectFill"
          />
        </View>

        <View className="user-info">
          <View className="user-info__name">
            {user.nickname}
            <Tag
              size="small"
              color={user.status === 1 ? 'success' : 'default'}
              className="user-info__status"
            >
              {user.status === 1 ? '启用' : '禁用'}
            </Tag>
          </View>

          <View className="user-info__username">
            @ {user.username}
          </View>

          <View className="user-info__meta">
            <Tag size="small" color={getGenderColor(user.genderLabel)}>
              {getGenderLabel(user.genderLabel)}
            </Tag>

            {user.deptName && (
              <Tag size="small" variant="outlined">
                {user.deptName}
              </Tag>
            )}
          </View>
        </View>
      </View>

      <View className="user-card__body">
        {user.mobile && (
          <View className="contact-item">
            <Phone className="contact-item__icon" />
            <Text className="contact-item__text">{user.mobile}</Text>
          </View>
        )}

        {user.email && (
          <View className="contact-item">
            <Phone className="contact-item__icon" />
            <Text className="contact-item__text">{user.email}</Text>
          </View>
        )}

        {user.roleNames && (
          <View className="roles-item">
            <View className="roles-label">角色：</View>
            <View className="roles-list">
              {user.roleNames.split(',').slice(0, 2).map((roleName, index) => (
                <Tag key={index} size="small" variant="outlined">
                  {roleName.trim()}
                </Tag>
              ))}
              {user.roleNames.split(',').length > 2 && (
                <Tag size="small" variant="outlined">
                  +{user.roleNames.split(',').length - 2}
                </Tag>
              )}
            </View>
          </View>
        )}
      </View>

      <View className="user-card__footer">
        <Space className="action-buttons">
          {hasPermission('sys:user:password:reset') && (
            <Button
              size="mini"
              variant="outlined"
              onClick={onResetPassword}
            >
              <Lock /> 重置密码
            </Button>
          )}

          {hasPermission('sys:user:edit') && (
            <Button
              size="mini"
              variant="outlined"
              onClick={onEdit}
            >
              <Edit /> 编辑
            </Button>
          )}

          {hasPermission('sys:user:delete') && (
            <Button
              size="mini"
              variant="outlined"
              color="danger"
              onClick={onDelete}
            >
              <Delete /> 删除
            </Button>
          )}
        </Space>
      </View>
    </View>
  );
};

export default UserCard;