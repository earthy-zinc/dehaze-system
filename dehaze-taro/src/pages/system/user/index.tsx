import React, { useState, useEffect } from 'react';
import { View } from '@tarojs/components';
import Taro, { usePageScroll } from '@tarojs/taro';
import {
  Navbar,
  Search,
  Button,
  Loading,
  Empty,
  PullRefresh,
  Toast
} from '@taroify/core';
import { Plus } from '@taroify/icons';
import UserCard from './components/UserCard';
import { useUserManagement } from '@/hooks/useUserManagement';
import { usePermission } from '@/hooks/usePermission';
import type { UserPageVO } from 'dehaze-sdk-js';
import './index.scss';

const UserListPage: React.FC = () => {
  const { hasPermission } = usePermission();
  const {
    users,
    loading,
    total,
    queryParams,
    fetchUsers,
    deleteUser,
    resetPassword,
    searchUsers,
  } = useUserManagement();

  const [searchValue, setSearchValue] = useState('');
  const [refreshing, setRefreshing] = useState(false);
  const [reachTop, setReachTop] = useState(true);

  // 页面滚动监听
  usePageScroll(({ scrollTop }) => setReachTop(scrollTop === 0));

  // 初始化数据
  useEffect(() => {
    fetchUsers();
  }, [fetchUsers]);

  // 下拉刷新
  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await fetchUsers({ pageNum: 1 });
    } finally {
      setRefreshing(false);
    }
  };

  // 上拉加载更多
  const handleLoadMore = async () => {
    if (users.length < total) {
      await fetchUsers({ pageNum: queryParams.pageNum + 1 });
    }
  };

  // 搜索
  const handleSearch = async (event: any) => {
    const value = event.detail.value || '';
    setSearchValue(value);
    if (value.trim()) {
      await searchUsers(value.trim());
    } else {
      await fetchUsers({ pageNum: 1 });
    }
  };

  // 删除用户
  const handleDelete = async (user: UserPageVO) => {
    const confirmed = await new Promise<boolean>((resolve) => {
      Taro.showModal({
        title: '确认删除',
        content: `确定要删除用户"${user.nickname}"吗？`,
        success: (res) => {
          if (res.confirm) {
            resolve(true);
          } else if (res.cancel) {
            resolve(false);
          }
        }
      });
    });

    if (confirmed && user.id) {
      try {
        await deleteUser(user.id);
      } catch (error) {
        // 错误已在 hook 中处理
      }
    }
  };

  // 重置密码
  const handleResetPassword = async (user: UserPageVO) => {
    const confirmed = await new Promise<boolean>((resolve) => {
      Taro.showModal({
        title: '重置密码',
        content: `确定要重置用户"${user.nickname}"的密码为默认密码"123456"吗？`,
        success: (res) => {
          if (res.confirm) {
            resolve(true);
          } else if (res.cancel) {
            resolve(false);
          }
        }
      });
    });

    if (confirmed && user.id) {
      try {
        const newPassword = '123456'; // 默认密码
        await resetPassword(user.id, newPassword);
      } catch (error) {
        // 错误已在 hook 中处理
      }
    } else {
      Toast.open({ message: '请输入新密码', position: 'top' });
    }
  };

  // 新增用户
  const handleAddUser = () => {
    Taro.navigateTo({
      url: '/pages/system/user/detail'
    });
  };

  // 编辑用户
  const handleEditUser = (userId: number) => {
    Taro.navigateTo({
      url: `/pages/system/user/detail?id=${userId}`
    });
  };

  return (
    <View className="user-list-page">
      {/* 导航栏 */}
      <Navbar title="用户管理">
        <Navbar.NavRight>
          {hasPermission('sys:user:add') && (
            <Button
              size="small"
              color="primary"
              onClick={handleAddUser}
            >
              <Plus /> 新增
            </Button>
          )}
        </Navbar.NavRight>
      </Navbar>

      {/* 搜索栏 */}
      <View className="search-bar">
        <Search
          placeholder="搜索用户名/昵称/手机号"
          value={searchValue}
          onChange={(e) => setSearchValue(e.detail.value)}
          onSearch={handleSearch}
          clearable
          action
        />
      </View>

      {/* 用户列表 */}
      <PullRefresh
        loading={refreshing}
        reachTop={reachTop}
        onRefresh={handleRefresh}
      >
        <PullRefresh.Completed>刷新成功</PullRefresh.Completed>
        <View className="user-list">
          {users.map((user) => (
            <UserCard
              key={user.id}
              user={user}
              onEdit={() => user.id && handleEditUser(user.id)}
              onDelete={() => handleDelete(user)}
              onResetPassword={() => handleResetPassword(user)}
            />
          ))}

          {loading && <Loading>加载中...</Loading>}

          {!loading && users.length === 0 && (
            <Empty>
              <Empty.Image />
              <Empty.Description>
                {searchValue ? '没有找到匹配的用户' : '暂无用户数据'}
              </Empty.Description>
            </Empty>
          )}
        </View>
      </PullRefresh>

          {/* 上拉加载提示 */}
      {users.length > 0 && users.length < total && !loading && (
        <View className="load-more-tip">
          <Button
            size="small"
            variant="outlined"
            onClick={handleLoadMore}
          >
            加载更多
          </Button>
        </View>
      )}

      {/* 列表为空时的提示 */}
      {!loading && total === 0 && (
        <View className="empty-state">
          <Empty>
            <Empty.Image />
            <Empty.Description>暂无用户数据</Empty.Description>
          </Empty>
          {hasPermission('sys:user:add') && (
            <Button
              color="primary"
              className="empty-state__button"
              onClick={handleAddUser}
            >
              新增第一个用户
            </Button>
          )}
        </View>
      )}
    </View>
  );
};

export default UserListPage;
