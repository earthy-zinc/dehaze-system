import React, { useState } from "react";
import { View } from "@tarojs/components";
import type { BaseEventOrig } from "@tarojs/components";
import Taro, {
  useLoad,
  usePullDownRefresh,
  useReachBottom,
} from "@tarojs/taro";
import {
  Navbar,
  Search,
  Button,
  Loading,
  Empty,
  SwipeCell,
  Cell,
} from "@taroify/core";
import { confirmDialog } from "@/utils/dialog";
import { formatDateTime } from "@/utils/format";
import { ArrowLeft, Add, Edit, Delete, Lock } from "@taroify/icons";
import { useRoleManagement } from "@/hooks/useRoleManagement";
import { usePermission } from "@/hooks/usePermission";
import ErrorState from "@/components/common/ErrorState";
import StatusTag from "@/components/common/StatusTag";
import "./index.less";

const RoleListPage: React.FC = () => {
  const {
    roles,
    loading,
    loadError,
    total,
    queryParams,
    fetchRoles,
    deleteRole,
    searchRoles,
    resetQuery,
  } = useRoleManagement();

  const { hasPermission } = usePermission();

  const [searchKeyword, setSearchKeyword] = useState("");

  // 页面加载
  useLoad(async () => {
    await fetchRoles();
  });

  // 下拉刷新
  usePullDownRefresh(async () => {
    try {
      await fetchRoles({ pageNum: 1 });
      Taro.stopPullDownRefresh();
    } catch (error) {
      Taro.stopPullDownRefresh();
    }
  });

  // 上拉加载更多
  useReachBottom(async () => {
    if (roles.length < total) {
      await fetchRoles({ pageNum: (queryParams.pageNum ?? 1) + 1 });
    }
  });

  // 搜索处理
  const performSearch = async (value: string) => {
    setSearchKeyword(value);
    if (value.trim()) {
      await searchRoles(value.trim());
    } else {
      await resetQuery();
    }
  };

  const handleSearch = async (event: BaseEventOrig<{ value: string }>) => {
    await performSearch(event.detail?.value || "");
  };

  // 新增角色
  const handleAdd = () => {
    Taro.navigateTo({
      url: "/pages/system/role/detail",
    });
  };

  // 编辑角色
  const handleEdit = (id: number | undefined) => {
    if (!id) return;
    Taro.navigateTo({
      url: `/pages/system/role/detail?id=${id}`,
    });
  };

  // 权限配置
  const handlePermission = (id: number | undefined) => {
    if (!id) return;
    Taro.navigateTo({
      url: `/pages/system/role/permission?id=${id}`,
    });
  };

  // 删除确认
  const handleDelete = async (role: any) => {
    const confirmed = await confirmDialog({
      title: "确认删除",
      content: `确定要删除角色 "${role.name}" 吗？此操作不可恢复。`,
      confirmText: "删除",
      cancelText: "取消",
    });
    if (!confirmed) return;
    await confirmDelete(role);
  };

  // 确认删除
  const confirmDelete = async (role: any) => {
    if (!role) return;
    try {
      await deleteRole(role.id);
    } catch (error) {
      // 错误已在 hook 中处理
    }
  };

  return (
    <View className="role-list-page">
      <Navbar title="角色管理">
        <Navbar.NavLeft>
          <ArrowLeft onClick={() => Taro.navigateBack()} />
        </Navbar.NavLeft>
        <Navbar.NavRight>
          {hasPermission("sys:role:add") && <Add onClick={handleAdd} />}
        </Navbar.NavRight>
      </Navbar>

      {/* 搜索栏 */}
      <View className="search-bar">
        <Search
          placeholder="请输入角色名称"
          value={searchKeyword}
          onChange={(e) => setSearchKeyword(e.detail.value)}
          onSearch={handleSearch}
          onClear={() => performSearch("")}
        />
      </View>

      {/* 角色列表 */}
      <View className="role-list">
        {loading && roles.length === 0 ? (
          <Loading>加载中...</Loading>
        ) : loadError && roles.length === 0 ? (
          <ErrorState message={loadError} onRetry={() => fetchRoles()} />
        ) : roles.length === 0 ? (
          <Empty>
            <Empty.Image />
            <Empty.Description>暂无角色数据</Empty.Description>
            {hasPermission("sys:role:add") && (
              <Button
                className="empty-state__button"
                color="primary"
                size="small"
                onClick={handleAdd}
              >
                新增角色
              </Button>
            )}
          </Empty>
        ) : (
          roles.map((role) => (
            <SwipeCell key={role.id} className="role-swipe-cell">
              <SwipeCell.Actions side="right">
                {hasPermission("sys:role:permission") && (
                  <Button
                    className="action-btn permission-btn"
                    size="small"
                    onClick={() => handlePermission(role.id)}
                  >
                    <Lock />
                    权限
                  </Button>
                )}
                {hasPermission("sys:role:edit") && (
                  <Button
                    className="action-btn edit-btn"
                    size="small"
                    onClick={() => handleEdit(role.id)}
                  >
                    <Edit />
                    编辑
                  </Button>
                )}
                {hasPermission("sys:role:delete") && (
                  <Button
                    className="action-btn delete-btn"
                    size="small"
                    onClick={() => handleDelete(role)}
                  >
                    <Delete />
                    删除
                  </Button>
                )}
              </SwipeCell.Actions>
              <Cell className="role-cell">
                <View className="role-info">
                  <View className="role-name">{role.name}</View>
                  <View className="role-code">编码: {role.code}</View>
                </View>
                <View className="role-status">
                  <StatusTag status={role.status} />
                </View>
                <View className="role-meta">
                  <View className="meta-item">
                    <View className="meta-label">排序:</View>
                    <View className="meta-value">{role.sort}</View>
                  </View>
                </View>
                {role.createTime && (
                  <View className="role-time">
                    创建时间: {formatDateTime(role.createTime, true)}
                  </View>
                )}
              </Cell>
            </SwipeCell>
          ))
        )}
      </View>

      {/* 加载更多 */}
      {loading && roles.length > 0 && (
        <View className="loading-more">
          <Loading size="small">加载中...</Loading>
        </View>
      )}
    </View>
  );
};

export default RoleListPage;
