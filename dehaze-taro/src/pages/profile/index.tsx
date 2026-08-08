import React, { useState, useEffect, useCallback, useMemo } from "react";
import { View, Text, ScrollView, Image } from "@tarojs/components";
import Taro from "@tarojs/taro";
import { Tag } from "@taroify/core";
import { MemberAPI, ModelAPI, FavoriteAPI, TaskAPI } from "dehaze-sdk-js";
import type { MemberProfileVO } from "dehaze-sdk-js";
import { confirmDialog } from "@/utils/dialog";
import { useAuth } from "@/hooks/useAuth";
import { usePermission } from "@/hooks/usePermission";
import { tabBarItems } from "@/config/menu";
import PageLayout from "@/layout";
import { getErrorMessage } from "@/utils/error";
import "./index.less";

// ==================== 类型定义 ====================

interface StatItem {
  label: string;
  value: string | number;
  route?: string;
}

interface EntryItem {
  icon: string;
  title: string;
  route: string;
  permission?: string;
}

interface EntryGroup {
  title: string;
  entries: EntryItem[];
}

// ==================== 入口分组配置 ====================

const PERSONAL_DATA_ENTRIES: EntryItem[] = [
  { icon: "📄", title: "我的文件", route: "/pages/personal/files/index" },
  { icon: "📊", title: "我的数据集", route: "/pages/dataset/index" },
  { icon: "⏱️", title: "处理历史", route: "/pages/task/index" },
  { icon: "⭐", title: "我的收藏", route: "/pages/favorite/index" },
];

const BUSINESS_ENTRIES: EntryItem[] = [
  { icon: "👑", title: "我的会员", route: "/pages/personal/member/index" },
  { icon: "📦", title: "我的套餐", route: "/pages/personal/package/index" },
  { icon: "🛒", title: "我的订单", route: "/pages/personal/orders/index" },
  { icon: "💰", title: "我的额度", route: "/pages/personal/quota/index" },
  { icon: "💬", title: "反馈评价", route: "/pages/personal/feedback/index" },
];

const OTHER_ENTRIES: EntryItem[] = [
  { icon: "⚙️", title: "系统设置", route: "/pages/personal/settings/index" },
  { icon: "❓", title: "帮助中心", route: "/pages/personal/help/index" },
  { icon: "ℹ️", title: "关于我们", route: "/pages/personal/about/index" },
  { icon: "🔔", title: "消息设置", route: "/pages/notify/index" },
];

// 管理入口分组（仅管理员/有权限用户可见，按 dev-admin 规划）
const ADMIN_GROUPS: EntryGroup[] = [
  {
    title: "工作台",
    entries: [
      {
        icon: "📊",
        title: "工作台",
        route: "/pages/dashboard/index",
        permission: "sys:user:*",
      },
    ],
  },
  {
    title: "用户与权限",
    entries: [
      {
        icon: "👥",
        title: "用户管理",
        route: "/pages/system/user/index",
        permission: "sys:user:*",
      },
      {
        icon: "🛡️",
        title: "角色管理",
        route: "/pages/system/role/index",
        permission: "sys:role:*",
      },
      {
        icon: "📑",
        title: "菜单管理",
        route: "/pages/system/menu/index",
        permission: "sys:menu:*",
      },
      {
        icon: "🏢",
        title: "部门管理",
        route: "/pages/system/dept/index",
        permission: "sys:dept:*",
      },
      {
        icon: "📚",
        title: "字典管理",
        route: "/pages/system/dict/index",
        permission: "sys:dict:*",
      },
    ],
  },
  {
    title: "算法与数据",
    entries: [
      {
        icon: "🧠",
        title: "算法管理",
        route: "/pages/system/algorithm/index",
        permission: "sys:algorithm:*",
      },
      {
        icon: "📊",
        title: "数据集管理",
        route: "/pages/system/dataset/index",
        permission: "sys:dataset:*",
      },
      {
        icon: "🎯",
        title: "推荐管理",
        route: "/pages/system/recommend/index",
        permission: "sys:recommendation:*",
      },
    ],
  },
  {
    title: "业务管理",
    entries: [
      {
        icon: "👑",
        title: "会员管理",
        route: "/pages/system/member/index",
        permission: "sys:member:*",
      },
      {
        icon: "📦",
        title: "套餐管理",
        route: "/pages/system/package/index",
        permission: "sys:package:*",
      },
      {
        icon: "🛒",
        title: "订单管理",
        route: "/pages/system/order/index",
        permission: "sys:order:*",
      },
    ],
  },
  {
    title: "运营管理",
    entries: [
      {
        icon: "⏱️",
        title: "任务管理",
        route: "/pages/system/task/index",
        permission: "sys:task:*",
      },
      {
        icon: "💬",
        title: "反馈评价管理",
        route: "/pages/system/feedback/index",
        permission: "sys:feedback:*",
      },
      {
        icon: "🔔",
        title: "消息管理",
        route: "/pages/system/message/index",
        permission: "sys:message:*",
      },
    ],
  },
];

// ==================== 页面组件 ====================

const ProfilePage: React.FC = () => {
  const { user, roles, logout } = useAuth();
  const { hasPermission } = usePermission();

  // 会员信息
  const [member, setMember] = useState<MemberProfileVO | null>(null);
  // 数据统计
  const [quotaCount, setQuotaCount] = useState<number | null>(null);
  const [favoriteCount, setFavoriteCount] = useState<number | null>(null);
  const [taskTotal, setTaskTotal] = useState<number | null>(null);

  // 加载会员信息
  const loadMember = useCallback(async () => {
    try {
      const profile = await MemberAPI.getProfile();
      setMember(profile);
    } catch {
      // 非会员静默处理
    }
  }, []);

  // 加载数据统计
  const loadStats = useCallback(async () => {
    try {
      const quota = await ModelAPI.getQuota();
      setQuotaCount(quota.remaining ?? quota.total ?? null);
    } catch {
      setQuotaCount(null);
    }
    try {
      const fav = await FavoriteAPI.getCount();
      if (fav && fav.length > 0) {
        const total = fav.reduce((sum, item) => sum + (item.count || 0), 0);
        setFavoriteCount(total);
      }
    } catch {
      setFavoriteCount(null);
    }
    try {
      const tasks = await TaskAPI.getPage({ pageNum: 1, pageSize: 1 });
      setTaskTotal(tasks.total ?? null);
    } catch {
      setTaskTotal(null);
    }
  }, []);

  useEffect(() => {
    loadMember();
    loadStats();
  }, [loadMember, loadStats]);

  // 数据统计项
  const stats: StatItem[] = useMemo(() => {
    const items: StatItem[] = [
      {
        label: "剩余额度",
        value: quotaCount !== null ? `${quotaCount} 次` : "-",
        route: "/pages/personal/quota/index",
      },
      {
        label: "处理次数",
        value: taskTotal !== null ? `${taskTotal}` : "-",
        route: "/pages/task/index",
      },
      {
        label: "我的收藏",
        value: favoriteCount !== null ? `${favoriteCount}` : "-",
        route: "/pages/favorite/index",
      },
    ];
    return items;
  }, [quotaCount, taskTotal, favoriteCount]);

  // 权限过滤后的管理入口分组
  const visibleAdminGroups = useMemo(
    () =>
      ADMIN_GROUPS.map((group) => ({
        ...group,
        entries: group.entries.filter((entry) =>
          entry.permission ? hasPermission(entry.permission) : true
        ),
      })).filter((group) => group.entries.length > 0),
    [hasPermission]
  );

  // 跳转
  const handleNavigate = useCallback((route: string) => {
    if (tabBarItems.some((item) => item.route === route)) {
      Taro.switchTab({ url: route });
    } else {
      Taro.navigateTo({
        url: route,
        fail: () => {
          Taro.showToast({ title: "页面跳转失败", icon: "none" });
        },
      });
    }
  }, []);

  // 退出登录
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
      setTimeout(() => {
        Taro.reLaunch({ url: "/pages/login/index" });
      }, 800);
    } catch (err: unknown) {
      Taro.showToast({ title: getErrorMessage(err, "退出失败"), icon: "none" });
    }
  }, [logout]);

  // 头像首字母
  const avatarLetter = user?.nickname?.[0] || user?.username?.[0] || "U";
  const isFullAvatar = (avatar?: string) =>
    !!avatar && /^https?:\/\//.test(avatar);

  // VIP 状态
  const isVip =
    !!member &&
    member.levelCode !== "level_1" &&
    member.levelCode !== "level_0";

  // ==================== 渲染 ====================

  const renderEntry = (entry: EntryItem) => (
    <View
      key={entry.route}
      className="entry-item"
      onClick={() => handleNavigate(entry.route)}
    >
      <Text className="entry-icon">{entry.icon}</Text>
      <Text className="entry-title">{entry.title}</Text>
      <Text className="entry-arrow">›</Text>
    </View>
  );

  const renderEntryGroup = (group: EntryGroup) => (
    <View key={group.title} className="entry-group-section">
      <Text className="group-title">{group.title}</Text>
      <View className="entry-card">{group.entries.map(renderEntry)}</View>
    </View>
  );

  return (
    <PageLayout level="L1" title="我的">
      <View className="profile-page">
        <ScrollView scrollY className="profile-scroll">
          {/* 用户卡 */}
          <View className="user-card">
            <View className="user-card-bg" />
            <View className="user-card-content">
              <View className="avatar-wrapper">
                {isFullAvatar(user?.avatar) ? (
                  <Image
                    className="avatar-img"
                    src={user!.avatar!}
                    mode="aspectFill"
                  />
                ) : (
                  <Text className="avatar-text">{avatarLetter}</Text>
                )}
              </View>
              <View className="user-info">
                <Text className="user-nickname">
                  {user?.nickname || "未设置昵称"}
                </Text>
                <View className="user-roles">
                  {roles.length > 0 ? (
                    roles.slice(0, 2).map((role) => (
                      <Tag key={role} size="small" color="primary">
                        {role.replace("ROLE_", "")}
                      </Tag>
                    ))
                  ) : (
                    <Text className="no-role">普通用户</Text>
                  )}
                </View>
              </View>
            </View>
          </View>

          {/* VIP 横幅 */}
          <View
            className="vip-banner"
            onClick={() => handleNavigate("/pages/personal/member/index")}
          >
            <View className="vip-banner-left">
              <Text className="vip-banner-icon">👑</Text>
              <View className="vip-banner-text">
                <Text className="vip-banner-title">
                  {isVip
                    ? `${member?.levelName || "会员"}专属权益`
                    : "开通 VIP 畅享更多次数"}
                </Text>
                <Text className="vip-banner-desc">
                  {isVip
                    ? `成长值 ${member?.growthValue || 0}，点击查看详情`
                    : "解锁全部高级功能"}
                </Text>
              </View>
            </View>
            <View className="vip-banner-action">
              <Text className="vip-banner-btn">
                {isVip ? "详情" : "去开通"}
              </Text>
              <Text className="vip-banner-arrow">›</Text>
            </View>
          </View>

          {/* 数据统计 */}
          <View className="stats-card">
            {stats.map((stat, idx) => (
              <View
                key={stat.label}
                className={`stat-item ${idx < stats.length - 1 ? "stat-divider" : ""}`}
                onClick={() => stat.route && handleNavigate(stat.route)}
              >
                <Text className="stat-value">{stat.value}</Text>
                <Text className="stat-label">{stat.label}</Text>
              </View>
            ))}
          </View>

          {/* 个人数据分组 */}
          <View className="entry-group-section">
            <Text className="group-title">个人数据</Text>
            <View className="entry-card">
              {PERSONAL_DATA_ENTRIES.map(renderEntry)}
            </View>
          </View>

          {/* 商业服务分组 */}
          <View className="entry-group-section">
            <Text className="group-title">商业服务</Text>
            <View className="entry-card">
              {BUSINESS_ENTRIES.map(renderEntry)}
            </View>
          </View>

          {/* 其他分组 */}
          <View className="entry-group-section">
            <Text className="group-title">其他</Text>
            <View className="entry-card">{OTHER_ENTRIES.map(renderEntry)}</View>
          </View>

          {/* 管理入口分组（权限过滤） */}
          {visibleAdminGroups.map(renderEntryGroup)}

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
