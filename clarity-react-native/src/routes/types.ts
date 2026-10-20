import type { NavigatorScreenParams } from '@react-navigation/native';
import type { EvaluationMetrics } from '@/types/evaluation';
import type { SelectedImage } from '@/types/image';
import type { InputMethod } from '@/pages/image-input/types/imageInput';

// ============================================================
// Compare 路由共享参数
// ============================================================
export interface CompareRouteParams {
  originalUrl: string;
  processedUrl: string;
  cleanUrl?: string;
  /** 图像模型标识（统一模型规范 slug），用于指标评估 */
  modelId?: string;
}

// ============================================================
// Auth Stack
// ============================================================
export type AuthStackParamList = {
  Login: undefined;
  Register: undefined;
};

// ============================================================
// Home Tab Stack
// ============================================================
export type HomeStackParamList = {
  Index: undefined;
};

// ============================================================
// Tools Tab Stack
// ============================================================
export type ToolsStackParamList = {
  Index: undefined;
  ImageInput: { initialMethod?: InputMethod } | undefined;
  Dataset: undefined;
  DatasetBrowse: undefined;
  DatasetDetail: { datasetId: number };
  Task: undefined;
  MetricsManage: undefined;
  Batch: undefined;
  Processing: { model?: string; image?: SelectedImage } | undefined;
};

// ============================================================
// Clarity Tab Stack
// ============================================================
export type DehazeStackParamList = {
  Index: undefined;
  Processing: { model?: string; image?: SelectedImage } | undefined;
  Batch: undefined;
  CompareSideBySide: CompareRouteParams | undefined;
  CompareOverlay: CompareRouteParams | undefined;
  CompareMagnifier: CompareRouteParams | undefined;
  CompareFilter: CompareRouteParams | undefined;
  CompareMetrics:
    | (CompareRouteParams & { metrics?: EvaluationMetrics })
    | undefined;
};

// ============================================================
// Messages Tab Stack
// ============================================================
export type MessagesStackParamList = {
  Index: undefined;
  MessageDetail: { messageId: number };
};

// ============================================================
// Profile Tab Stack (dev-personal + dev-admin)
// ============================================================
export type ProfileStackParamList = {
  Index: undefined;
  // ===== 个人 L2 (dev-personal) =====
  PersonalFiles: undefined;
  PersonalOrders: undefined;
  PersonalQuota: undefined;
  PersonalMember: undefined;
  PersonalPackage: undefined;
  PersonalFeedback: undefined;
  PersonalFavorites: undefined;
  PersonalSettings: undefined;
  PersonalHelp: undefined;
  PersonalAbout: undefined;
  Notify: undefined;
  // Task 归位（处理历史）
  Task: undefined;
  Dataset: undefined;
  DatasetDetail: { datasetId: number };
  // ===== 管理入口 (dev-admin) =====
  SystemDashboard: undefined;
  SystemUser: undefined;
  SystemUserForm: { userId?: number } | undefined;
  SystemRole: undefined;
  SystemRoleForm: { roleId?: number } | undefined;
  SystemRolePerm: { roleId: number };
  SystemMenu: undefined;
  SystemMenuForm: { menuId?: number } | undefined;
  SystemDept: undefined;
  SystemDeptForm: { deptId?: number } | undefined;
  SystemDict: undefined;
  SystemDictTypeForm: { dictTypeId?: number } | undefined;
  SystemDictItem: { typeCode: string; typeName: string };
  SystemDictItemForm: { dictItemId?: number; typeCode: string } | undefined;
  SystemDataset: undefined;
  SystemDatasetForm: { datasetId?: number } | undefined;
  SystemTask: undefined;
  SystemMember: undefined;
  SystemMemberDetail: { userId: number };
  SystemMemberGrowthLog: { userId: number };
  SystemPackage: undefined;
  SystemPackageForm: { packageId?: number } | undefined;
  SystemOrder: undefined;
  SystemOrderDetail: { orderNo: string };
  SystemOrderRefund: undefined;
  SystemFeedback: undefined;
  SystemFeedbackDetail: { feedbackId: number };
  SystemMessage: undefined;
  SystemMessageAnnouncement: undefined;
  SystemMessageAnnouncementForm: { announcementId?: number } | undefined;
  SystemMessageTemplate: undefined;
  SystemMessageTemplateForm: { templateId?: number } | undefined;
  SystemMessageSend: undefined;
  SystemRecommend: undefined;
  SystemRecommendRuleForm: { ruleId?: number } | undefined;
};

// ============================================================
// Bottom Tab ParamList
// ============================================================
export type TabParamList = {
  Home: NavigatorScreenParams<HomeStackParamList> | undefined;
  Tools: NavigatorScreenParams<ToolsStackParamList> | undefined;
  Dehaze: NavigatorScreenParams<DehazeStackParamList> | undefined;
  Messages: NavigatorScreenParams<MessagesStackParamList> | undefined;
  Profile: NavigatorScreenParams<ProfileStackParamList> | undefined;
};
