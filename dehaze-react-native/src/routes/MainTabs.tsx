/**
 * MainTabs — 底部 5 Tab 导航
 *
 * 每个 Tab 内部嵌套 NativeStack，形成 L1 Tab + L2/L3 Stack 结构。
 * 使用内置 BottomTabBar，不自定义渲染函数。
 */
import React from 'react';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import Ionicons from 'react-native-vector-icons/Ionicons';
import { colors } from '@/theme/colors';

import type {
  TabParamList,
  HomeStackParamList,
  ToolsStackParamList,
  DehazeStackParamList,
  MessagesStackParamList,
  ProfileStackParamList,
} from './types';

// 页面导入
import HomeScreen from '@/pages/home';
import ToolsScreen from '@/pages/tools';
import DehazeScreen from '@/pages/dehaze';
import MessagesScreen from '@/pages/messages';
import ProfileScreen from '@/pages/profile';
import ImageInputScreen from '@/pages/image-input';
import AlgorithmSelectScreen from '@/pages/algorithm-select';
import AlgorithmBrowseScreen from '@/pages/algorithm-browse';
import AlgorithmScreen from '@/pages/algorithm';
import DatasetScreen from '@/pages/dataset';
import DatasetBrowseScreen from '@/pages/dataset-browse';
import BatchScreen from '@/pages/batch';
import MetricsManageScreen from '@/pages/metrics-manage';
import TaskScreen from '@/pages/task';
import ProcessingScreen from '@/pages/processing';
import SideBySideScreen from '@/pages/compare/SideBySide';
import OverlayScreen from '@/pages/compare/Overlay';
import MagnifierScreen from '@/pages/compare/Magnifier';
import FilterScreen from '@/pages/compare/Filter';
import MetricsScreen from '@/pages/compare/Metrics';
// ===== 个人 L2 (dev-personal) =====
import PersonalFilesScreen from '@/pages/personal/files';
import PersonalOrdersScreen from '@/pages/personal/orders';
import PersonalQuotaScreen from '@/pages/personal/quota';
import PersonalMemberScreen from '@/pages/personal/member';
import PersonalPackageScreen from '@/pages/personal/package';
import PersonalFeedbackScreen from '@/pages/personal/feedback';
import PersonalFavoritesScreen from '@/pages/personal/favorites';
import PersonalSettingsScreen from '@/pages/personal/settings';
import PersonalHelpScreen from '@/pages/personal/help';
import PersonalAboutScreen from '@/pages/personal/about';
import NotifyScreen from '@/pages/notify';

const Tab = createBottomTabNavigator<TabParamList>();

const makeTabBarIcon = (activeIcon: string, inactiveIcon: string) =>
  function TabBarIcon({ color, focused }: { color: string; focused: boolean }) {
    return <Ionicons name={focused ? activeIcon : inactiveIcon} size={22} color={color} />;
  };

// ============================================================
// Stack Navigators
// ============================================================

const HomeStack = createNativeStackNavigator<HomeStackParamList>();
function HomeStackNavigator() {
  return (
    <HomeStack.Navigator screenOptions={{ headerShown: false }}>
      <HomeStack.Screen name="Index" component={HomeScreen} />
    </HomeStack.Navigator>
  );
}

const ToolsStack = createNativeStackNavigator<ToolsStackParamList>();
function ToolsStackNavigator() {
  return (
    <ToolsStack.Navigator screenOptions={{ headerShown: false }}>
      <ToolsStack.Screen name="Index" component={ToolsScreen} />
      <ToolsStack.Screen name="ImageInput" component={ImageInputScreen} />
      <ToolsStack.Screen name="AlgorithmSelect" component={AlgorithmSelectScreen} />
      <ToolsStack.Screen name="AlgorithmBrowse" component={AlgorithmBrowseScreen} />
      <ToolsStack.Screen name="Algorithm" component={AlgorithmScreen} />
      <ToolsStack.Screen name="Dataset" component={DatasetScreen} />
      <ToolsStack.Screen name="DatasetBrowse" component={DatasetBrowseScreen} />
      <ToolsStack.Screen name="Task" component={TaskScreen} />
      <ToolsStack.Screen name="MetricsManage" component={MetricsManageScreen} />
      <ToolsStack.Screen name="Batch" component={BatchScreen} />
      <ToolsStack.Screen name="Processing" component={ProcessingScreen} />
    </ToolsStack.Navigator>
  );
}

const DehazeStack = createNativeStackNavigator<DehazeStackParamList>();
function DehazeStackNavigator() {
  return (
    <DehazeStack.Navigator screenOptions={{ headerShown: false }}>
      <DehazeStack.Screen name="Index" component={DehazeScreen} />
      <DehazeStack.Screen name="AlgorithmSelect" component={AlgorithmSelectScreen} />
      <DehazeStack.Screen name="AlgorithmBrowse" component={AlgorithmBrowseScreen} />
      <DehazeStack.Screen name="Algorithm" component={AlgorithmScreen} />
      <DehazeStack.Screen name="Processing" component={ProcessingScreen} />
      <DehazeStack.Screen name="Batch" component={BatchScreen} />
      <DehazeStack.Screen
        name="CompareSideBySide"
        component={SideBySideScreen}
        options={{ headerShown: false }}
      />
      <DehazeStack.Screen
        name="CompareOverlay"
        component={OverlayScreen}
        options={{ headerShown: false }}
      />
      <DehazeStack.Screen
        name="CompareMagnifier"
        component={MagnifierScreen}
        options={{ headerShown: false }}
      />
      <DehazeStack.Screen
        name="CompareFilter"
        component={FilterScreen}
        options={{ headerShown: false }}
      />
      <DehazeStack.Screen
        name="CompareMetrics"
        component={MetricsScreen}
        options={{ headerShown: false }}
      />
    </DehazeStack.Navigator>
  );
}

const MessagesStack = createNativeStackNavigator<MessagesStackParamList>();
function MessagesStackNavigator() {
  return (
    <MessagesStack.Navigator screenOptions={{ headerShown: false }}>
      <MessagesStack.Screen name="Index" component={MessagesScreen} />
    </MessagesStack.Navigator>
  );
}

// ===== 管理模块 (dev-admin) =====
import SystemDashboardScreen from '@/pages/dashboard';
import SystemUserScreen from '@/pages/system/user';
import SystemUserFormScreen from '@/pages/system/user/form';
import SystemRoleScreen from '@/pages/system/role';
import SystemRoleFormScreen from '@/pages/system/role/form';
import SystemRolePermScreen from '@/pages/system/role/perm';
import SystemMenuScreen from '@/pages/system/menu';
import SystemMenuFormScreen from '@/pages/system/menu/form';
import SystemDeptScreen from '@/pages/system/dept';
import SystemDeptFormScreen from '@/pages/system/dept/form';
import SystemDictScreen from '@/pages/system/dict';
import SystemDictTypeFormScreen from '@/pages/system/dict/type-form';
import SystemDictItemScreen from '@/pages/system/dict/items';
import SystemDictItemFormScreen from '@/pages/system/dict/item-form';
import SystemAlgorithmScreen from '@/pages/system/algorithm';
import SystemAlgorithmFormScreen from '@/pages/system/algorithm/form';
import SystemAlgorithmAuditScreen from '@/pages/system/algorithm/audit';
import SystemDatasetScreen from '@/pages/system/dataset';
import SystemDatasetFormScreen from '@/pages/system/dataset/form';
import SystemTaskScreen from '@/pages/system/task';
import SystemMemberScreen from '@/pages/system/member';
import SystemMemberDetailScreen from '@/pages/system/member/detail';
import SystemMemberGrowthLogScreen from '@/pages/system/member/growth-log';
import SystemPackageScreen from '@/pages/system/package';
import SystemPackageFormScreen from '@/pages/system/package/form';
import SystemOrderScreen from '@/pages/system/order';
import SystemOrderDetailScreen from '@/pages/system/order/detail';
import SystemOrderRefundScreen from '@/pages/system/order/refund';
import SystemFeedbackScreen from '@/pages/system/feedback';
import SystemFeedbackDetailScreen from '@/pages/system/feedback/detail';
import SystemMessageScreen from '@/pages/system/message';
import SystemMessageAnnouncementScreen from '@/pages/system/message/announcement';
import SystemMessageAnnouncementFormScreen from '@/pages/system/message/announcement-form';
import SystemMessageTemplateScreen from '@/pages/system/message/template';
import SystemMessageTemplateFormScreen from '@/pages/system/message/template-form';
import SystemMessageSendScreen from '@/pages/system/message/send';
import SystemRecommendScreen from '@/pages/system/recommend';
import SystemRecommendRuleFormScreen from '@/pages/system/recommend/rule-form';

const ProfileStack = createNativeStackNavigator<ProfileStackParamList>();
function ProfileStackNavigator() {
  return (
    <ProfileStack.Navigator screenOptions={{ headerShown: false }}>
      <ProfileStack.Screen name="Index" component={ProfileScreen} />
      {/* ===== 个人 L2 (dev-personal) ===== */}
      <ProfileStack.Screen name="PersonalFiles" component={PersonalFilesScreen} />
      <ProfileStack.Screen name="PersonalOrders" component={PersonalOrdersScreen} />
      <ProfileStack.Screen name="PersonalQuota" component={PersonalQuotaScreen} />
      <ProfileStack.Screen name="PersonalMember" component={PersonalMemberScreen} />
      <ProfileStack.Screen name="PersonalPackage" component={PersonalPackageScreen} />
      <ProfileStack.Screen name="PersonalFeedback" component={PersonalFeedbackScreen} />
      <ProfileStack.Screen name="PersonalFavorites" component={PersonalFavoritesScreen} />
      <ProfileStack.Screen name="PersonalSettings" component={PersonalSettingsScreen} />
      <ProfileStack.Screen name="PersonalHelp" component={PersonalHelpScreen} />
      <ProfileStack.Screen name="PersonalAbout" component={PersonalAboutScreen} />
      <ProfileStack.Screen name="Notify" component={NotifyScreen} />
      {/* Task + Dataset 归位 */}
      <ProfileStack.Screen name="Task" component={TaskScreen} />
      <ProfileStack.Screen name="Dataset" component={DatasetScreen} />
      {/* ===== 管理入口 (dev-admin) ===== */}
      <ProfileStack.Screen name="SystemDashboard" component={SystemDashboardScreen} />
      <ProfileStack.Screen name="SystemUser" component={SystemUserScreen} />
      <ProfileStack.Screen name="SystemUserForm" component={SystemUserFormScreen} />
      <ProfileStack.Screen name="SystemRole" component={SystemRoleScreen} />
      <ProfileStack.Screen name="SystemRoleForm" component={SystemRoleFormScreen} />
      <ProfileStack.Screen name="SystemRolePerm" component={SystemRolePermScreen} />
      <ProfileStack.Screen name="SystemMenu" component={SystemMenuScreen} />
      <ProfileStack.Screen name="SystemMenuForm" component={SystemMenuFormScreen} />
      <ProfileStack.Screen name="SystemDept" component={SystemDeptScreen} />
      <ProfileStack.Screen name="SystemDeptForm" component={SystemDeptFormScreen} />
      <ProfileStack.Screen name="SystemDict" component={SystemDictScreen} />
      <ProfileStack.Screen name="SystemDictTypeForm" component={SystemDictTypeFormScreen} />
      <ProfileStack.Screen name="SystemDictItem" component={SystemDictItemScreen} />
      <ProfileStack.Screen name="SystemDictItemForm" component={SystemDictItemFormScreen} />
      <ProfileStack.Screen name="SystemAlgorithm" component={SystemAlgorithmScreen} />
      <ProfileStack.Screen name="SystemAlgorithmForm" component={SystemAlgorithmFormScreen} />
      <ProfileStack.Screen name="SystemAlgorithmAudit" component={SystemAlgorithmAuditScreen} />
      <ProfileStack.Screen name="SystemDataset" component={SystemDatasetScreen} />
      <ProfileStack.Screen name="SystemDatasetForm" component={SystemDatasetFormScreen} />
      <ProfileStack.Screen name="SystemTask" component={SystemTaskScreen} />
      <ProfileStack.Screen name="SystemMember" component={SystemMemberScreen} />
      <ProfileStack.Screen name="SystemMemberDetail" component={SystemMemberDetailScreen} />
      <ProfileStack.Screen name="SystemMemberGrowthLog" component={SystemMemberGrowthLogScreen} />
      <ProfileStack.Screen name="SystemPackage" component={SystemPackageScreen} />
      <ProfileStack.Screen name="SystemPackageForm" component={SystemPackageFormScreen} />
      <ProfileStack.Screen name="SystemOrder" component={SystemOrderScreen} />
      <ProfileStack.Screen name="SystemOrderDetail" component={SystemOrderDetailScreen} />
      <ProfileStack.Screen name="SystemOrderRefund" component={SystemOrderRefundScreen} />
      <ProfileStack.Screen name="SystemFeedback" component={SystemFeedbackScreen} />
      <ProfileStack.Screen name="SystemFeedbackDetail" component={SystemFeedbackDetailScreen} />
      <ProfileStack.Screen name="SystemMessage" component={SystemMessageScreen} />
      <ProfileStack.Screen name="SystemMessageAnnouncement" component={SystemMessageAnnouncementScreen} />
      <ProfileStack.Screen name="SystemMessageAnnouncementForm" component={SystemMessageAnnouncementFormScreen} />
      <ProfileStack.Screen name="SystemMessageTemplate" component={SystemMessageTemplateScreen} />
      <ProfileStack.Screen name="SystemMessageTemplateForm" component={SystemMessageTemplateFormScreen} />
      <ProfileStack.Screen name="SystemMessageSend" component={SystemMessageSendScreen} />
      <ProfileStack.Screen name="SystemRecommend" component={SystemRecommendScreen} />
      <ProfileStack.Screen name="SystemRecommendRuleForm" component={SystemRecommendRuleFormScreen} />
    </ProfileStack.Navigator>
  );
}

// ============================================================
// Tab Navigator
// ============================================================

export default function MainTabs() {
  return (
    <Tab.Navigator
      screenOptions={{
        headerShown: false,
        tabBarActiveTintColor: colors.primary,
        tabBarInactiveTintColor: colors.text.tertiary,
        tabBarLabelStyle: { fontSize: 11 },
        tabBarStyle: { backgroundColor: colors.background.primary, borderTopColor: colors.border.light },
      }}
    >
      <Tab.Screen
        name="Home"
        component={HomeStackNavigator}
        options={{ tabBarIcon: makeTabBarIcon('home', 'home-outline') }}
      />
      <Tab.Screen
        name="Tools"
        component={ToolsStackNavigator}
        options={{ tabBarIcon: makeTabBarIcon('grid', 'grid-outline') }}
      />
      <Tab.Screen
        name="Dehaze"
        component={DehazeStackNavigator}
        options={{ tabBarIcon: makeTabBarIcon('color-wand', 'color-wand-outline') }}
      />
      <Tab.Screen
        name="Messages"
        component={MessagesStackNavigator}
        options={{ tabBarIcon: makeTabBarIcon('notifications', 'notifications-outline') }}
      />
      <Tab.Screen
        name="Profile"
        component={ProfileStackNavigator}
        options={{ tabBarIcon: makeTabBarIcon('person', 'person-outline') }}
      />
    </Tab.Navigator>
  );
}
