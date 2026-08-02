import React, { useCallback, useMemo } from "react";
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

/** 个人中心入口项 */
interface ProfileEntry {
  icon: string;
  title: string;
  desc: string;
  route: string;
  /** 系统模块名，拥有该模块任一权限（sys:{module}:*）时才显示。
   *  设为 "*" 表示拥有任意 sys:* 权限时显示 */
  sysModule?: string;
}

/** 普通用户功能入口（登录即可见） */
const USER_ENTRIES: ProfileEntry[] = [
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
  {
    icon: "⭐",
    title: "我的收藏",
    desc: "查看收藏的内容",
    route: "/pages/favorite/index",
  },
  {
    icon: "📦",
    title: "套餐管理",
    desc: "查看可用套餐",
    route: "/pages/package/index",
  },
  {
    icon: "🔔",
    title: "消息通知",
    desc: "查看通知消息",
    route: "/pages/notify/index",
  },
  {
    icon: "👑",
    title: "会员管理",
    desc: "会员信息与权益",
    route: "/pages/member/index",
  },
  {
    icon: "💬",
    title: "反馈评价",
    desc: "提交使用反馈",
    route: "/pages/feedback/index",
  },
];

/** 系统管理入口（需对应模块权限） */
const ADMIN_ENTRIES: ProfileEntry[] = [
  {
    icon: "📊",
    title: "工作台",
    desc: "统计概览与管理总览",
    route: "/pages/dashboard/index",
    sysModule: "*",
  },
  {
    icon: "👥",
    title: "用户管理",
    desc: "管理用户账号",
    route: "/pages/system/user/index",
    sysModule: "user",
  },
  {
    icon: "🛡️",
    title: "角色管理",
    desc: "管理角色与权限",
    route: "/pages/system/role/index",
    sysModule: "role",
  },
  {
    icon: "📚",
    title: "字典管理",
    desc: "管理字典类型与数据",
    route: "/pages/system/dict/index",
    sysModule: "dict",
  },
  {
    icon: "📑",
    title: "菜单管理",
    desc: "管理菜单与路由",
    route: "/pages/system/menu/index",
    sysModule: "menu",
  },
  {
    icon: "🏢",
    title: "部门管理",
    desc: "管理组织部门",
    route: "/pages/system/dept/index",
    sysModule: "dept",
  },
  {
    icon: "🎯",
    title: "推荐规则",
    desc: "管理算法推荐规则",
    route: "/pages/recommend/index",
    sysModule: "recommendation",
  },
];

// ==================== 页面组件 ====================

const ProfilePage: React.FC = () => {
  const { user, roles, perms, logout } = useAuth();

  /** 按模块权限过滤后的系统管理入口 */
  const visibleAdminEntries = useMemo(
    () =>
      ADMIN_ENTRIES.filter((entry) => {
        if (!entry.sysModule) return true;
        if (entry.sysModule === "*")
          return perms.some((p) => p.startsWith("sys:"));
        return perms.some((p) => p.startsWith(`sys:${entry.sysModule}:`));
      }),
    [perms]
  );

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
  const renderEntry = (entry: ProfileEntry) => (
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
              <Text className="section-count">共 {perms.length} 项</Text>
            </View>
            <View className="permission-card">
              {perms.length > 0 ? (
                <>
                  <View className="permission-tags">
                    {perms.slice(0, MAX_PERMISSION_PREVIEW).map((perm) => (
                      <Tag key={perm} size="small">
                        {perm}
                      </Tag>
                    ))}
                  </View>
                  {perms.length > MAX_PERMISSION_PREVIEW && (
                    <Text className="permission-more">
                      等共 {perms.length} 项权限
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
            <View className="entry-list">{USER_ENTRIES.map(renderEntry)}</View>
          </View>

          {/* 系统管理（仅有对应模块权限时显示） */}
          {visibleAdminEntries.length > 0 && (
            <View className="section">
              <Text className="section-title">系统管理</Text>
              <View className="entry-list">
                {visibleAdminEntries.map(renderEntry)}
              </View>
            </View>
          )}

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
