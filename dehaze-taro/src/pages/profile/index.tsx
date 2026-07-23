import React, { useCallback } from "react";
import { View, Text, ScrollView, Image } from "@tarojs/components";
import Taro from "@tarojs/taro";
import { Tag } from "@taroify/core";
import { confirmDialog } from "@/utils/dialog";
import { useAuth } from "@/hooks/useAuth";
import { isTabBarPage } from "@/config/menu";
import PageLayout from "@/layout";
import { getErrorMessage } from "@/utils/error";
import "./index.less";

// ==================== 常量定义 ====================

/** 权限概览最多展示的权限数量 */
const MAX_PERMISSION_PREVIEW = 6;

/** 个人中心功能入口 */
const PROFILE_ENTRIES = [
  {
    icon: "📋",
    title: "处理历史",
    desc: "查看去雾任务记录",
    route: "/pages/task/index",
  },
  {
    icon: "🖼️",
    title: "图像输入历史",
    desc: "查看输入图片记录",
    route: "/pages/image-input/index",
  },
];

// ==================== 页面组件 ====================

const ProfilePage: React.FC = () => {
  const { user, roles, permissions, logout } = useAuth();

  /** 跳转到指定页面（tabbar 页面用 reLaunch，其余用 navigateTo） */
  const handleNavigate = useCallback((route: string) => {
    if (isTabBarPage(route)) {
      Taro.reLaunch({ url: route });
    } else {
      Taro.navigateTo({
        url: route,
        fail: () => {
          Taro.showToast({ title: "页面跳转失败", icon: "none" });
        },
      });
    }
  }, []);

  /** 退出登录（二次确认） */
  const handleLogout = useCallback(async () => {
    const confirmed = await confirmDialog({
      title: "退出登录",
      content: "确认退出当前账号吗？",
      confirmColor: "#ff4d4f",
    });
    if (!confirmed) return;
    try {
      await logout();
      Taro.showToast({ title: "已退出登录", icon: "success" });
      // 延迟跳转，让用户看到提示。使用 reLaunch 清空页面栈
      setTimeout(() => {
        Taro.reLaunch({ url: "/pages/login/index" });
      }, 800);
    } catch (err: unknown) {
      Taro.showToast({ title: getErrorMessage(err, "退出失败"), icon: "none" });
    }
  }, [logout]);

  /** 获取头像首字母（用于无头像时的占位） */
  const getAvatarLetter = (): string => {
    return user?.nickname?.[0] || user?.username?.[0] || "U";
  };

  /** 是否为完整的头像URL */
  const isFullAvatarUrl = (avatar?: string): boolean => {
    return !!avatar && /^https?:\/\//.test(avatar);
  };

  // ==================== 渲染 ====================

  /** 渲染功能入口项 */
  const renderEntry = (entry: (typeof PROFILE_ENTRIES)[number]) => (
    <View
      key={entry.route}
      className="entry-item"
      onClick={() => handleNavigate(entry.route)}
    >
      <Text className="entry-icon">{entry.icon}</Text>
      <View className="entry-text">
        <Text className="entry-title">{entry.title}</Text>
        <Text className="entry-desc">{entry.desc}</Text>
      </View>
      <Text className="entry-arrow">›</Text>
    </View>
  );

  return (
    <PageLayout showTabbar currentRoute="/pages/profile/index" title="个人中心">
      <View className="profile-page">
        <ScrollView scrollY className="profile-scroll">
          {/* 用户信息头部 */}
          <View className="profile-header">
            <View className="avatar-wrapper">
              {isFullAvatarUrl(user?.avatar) ? (
                <Image
                  className="avatar-img"
                  src={user!.avatar!}
                  mode="aspectFill"
                />
              ) : (
                <Text className="avatar-text">{getAvatarLetter()}</Text>
              )}
            </View>
            <View className="user-meta">
              <Text className="user-nickname">
                {user?.nickname || "未设置昵称"}
              </Text>
              <Text className="user-username">
                账号：{user?.username || "-"}
              </Text>
              <View className="user-roles">
                {roles.length > 0 ? (
                  roles.slice(0, 3).map((role) => (
                    <Tag key={role} size="small" color="primary">
                      {role.replace("ROLE_", "")}
                    </Tag>
                  ))
                ) : (
                  <Text className="no-role-text">暂无角色</Text>
                )}
              </View>
            </View>
          </View>

          {/* 权限概览 */}
          <View className="section">
            <View className="section-header">
              <Text className="section-title">权限概览</Text>
              <Text className="section-count">共 {permissions.length} 项</Text>
            </View>
            <View className="permission-card">
              {permissions.length > 0 ? (
                <>
                  <View className="permission-tags">
                    {permissions
                      .slice(0, MAX_PERMISSION_PREVIEW)
                      .map((perm) => (
                        <Tag key={perm} size="small">
                          {perm}
                        </Tag>
                      ))}
                  </View>
                  {permissions.length > MAX_PERMISSION_PREVIEW && (
                    <Text className="permission-more">
                      等共 {permissions.length} 项权限
                    </Text>
                  )}
                </>
              ) : (
                <Text className="permission-empty">暂无权限</Text>
              )}
            </View>
          </View>

          {/* 功能入口 */}
          <View className="section">
            <Text className="section-title">功能入口</Text>
            <View className="entry-list">
              {PROFILE_ENTRIES.map(renderEntry)}
            </View>
          </View>

          {/* 退出登录 */}
          <View className="logout-section">
            <View className="logout-btn" onClick={handleLogout}>
              <Text>退出登录</Text>
            </View>
          </View>

          {/* 页脚 */}
          <View className="profile-footer">
            <Text>图像去雾系统 v1.0</Text>
          </View>
        </ScrollView>
      </View>
    </PageLayout>
  );
};

export default ProfilePage;
