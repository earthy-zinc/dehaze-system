import 'package:dio/dio.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../core/auth/auth_error_handler.dart';
import '../core/network/api_client.dart';
import '../core/storage/token_storage.dart';
import '../services/algorithm_service.dart';
import '../services/auth_service.dart';
import '../services/evaluation_service.dart';
import '../services/file_service.dart';

import '../services/message_service.dart';
import '../services/dataset_service.dart';
import '../services/member_service.dart';
import '../services/order_service.dart';
import '../services/prediction_service.dart';
import '../services/package_service.dart';
import '../services/favorite_service.dart';
import '../services/feedback_service.dart';
import '../services/recommendation_service.dart';
import '../services/dict_service.dart';
import '../services/image_input_service.dart';
import '../services/import_export_service.dart';
import '../services/api_key_service.dart';
import '../services/announcement_service.dart';
import '../services/message_template_service.dart';
import '../services/notification_settings_service.dart';
import '../services/task_service.dart';
import '../services/role_service.dart';
import '../services/menu_service.dart';
import '../services/dept_service.dart';
import '../services/user_service.dart';

// ==================== 基础设施 Providers ====================

/// SharedPreferences Provider（必须在 main.dart 中 override）
final sharedPreferencesProvider = Provider<SharedPreferences>((ref) {
  throw UnimplementedError(
    'SharedPreferences must be initialized in main.dart',
  );
});

/// Token 存储 Provider
final tokenStorageProvider = Provider<TokenStorage>((ref) {
  final prefs = ref.watch(sharedPreferencesProvider);
  return TokenStorage(prefs);
});

/// 认证错误回调 Provider
///
/// 使用 AuthErrorHandler 静态容器，避免 Provider 循环依赖。
/// 在 DehazeApp 初始化时通过 AuthErrorHandler.setHandler 设置实际回调。
final authErrorCallbackProvider = Provider<void Function()>((ref) {
  return AuthErrorHandler.handle;
});

/// Dio Provider
final dioClientProvider = Provider<Dio>((ref) {
  final tokenStorage = ref.watch(tokenStorageProvider);
  final onAuthError = ref.watch(authErrorCallbackProvider);
  final apiClient = ApiClient.create(
    tokenStorage: tokenStorage,
    onAuthError: onAuthError,
  );
  return apiClient.dio;
});

// ==================== 服务 Providers ====================

final authServiceProvider = Provider<AuthService>((ref) {
  return AuthService(ref.watch(dioClientProvider));
});

final predictionServiceProvider = Provider<PredictionService>((ref) {
  return PredictionService(ref.watch(dioClientProvider));
});

final evaluationServiceProvider = Provider<EvaluationService>((ref) {
  return EvaluationService(ref.watch(dioClientProvider));
});

final fileServiceProvider = Provider<FileService>((ref) {
  return FileService(ref.watch(dioClientProvider));
});

final algorithmServiceProvider = Provider<AlgorithmService>((ref) {
  return AlgorithmService(ref.watch(dioClientProvider));
});

final favoriteServiceProvider = Provider<FavoriteService>((ref) {
  return FavoriteService(ref.watch(dioClientProvider));
});

final feedbackServiceProvider = Provider<FeedbackService>((ref) {
  return FeedbackService(ref.watch(dioClientProvider));
});

final recommendationServiceProvider = Provider<RecommendationService>((ref) {
  return RecommendationService(ref.watch(dioClientProvider));
});

final dictServiceProvider = Provider<DictService>((ref) {
  return DictService(ref.watch(dioClientProvider));
});

final orderServiceProvider = Provider<OrderService>((ref) {
  return OrderService(ref.watch(dioClientProvider));
});

final memberServiceProvider = Provider<MemberService>((ref) {
  return MemberService(ref.watch(dioClientProvider));
});

final messageServiceProvider = Provider<MessageService>((ref) {
  return MessageService(ref.watch(dioClientProvider));
});

final taskServiceProvider = Provider<TaskService>((ref) {
  return TaskService(ref.watch(dioClientProvider));
});

// ==================== 系统管理 Providers ====================

final userServiceProvider = Provider<UserService>((ref) {
  return UserService(ref.watch(dioClientProvider));
});

final roleServiceProvider = Provider<RoleService>((ref) {
  return RoleService(ref.watch(dioClientProvider));
});

final menuServiceProvider = Provider<MenuService>((ref) {
  return MenuService(ref.watch(dioClientProvider));
});

final deptServiceProvider = Provider<DeptService>((ref) {
  return DeptService(ref.watch(dioClientProvider));
});

final packageServiceProvider = Provider<PackageService>((ref) {
  return PackageService(ref.watch(dioClientProvider));
});

// ==================== 消息系统 Providers ====================

final announcementServiceProvider = Provider<AnnouncementService>((ref) {
  return AnnouncementService(ref.watch(dioClientProvider));
});

final messageTemplateServiceProvider = Provider<MessageTemplateService>((ref) {
  return MessageTemplateService(ref.watch(dioClientProvider));
});

final notificationSettingsServiceProvider = Provider<NotificationSettingsService>((ref) {
  return NotificationSettingsService(ref.watch(dioClientProvider));
});

// ==================== 数据集服务 Providers ====================

final datasetServiceProvider = Provider<DatasetService>((ref) {
  return DatasetService(ref.watch(dioClientProvider));
});

final datasetItemServiceProvider = Provider<DatasetItemService>((ref) {
  return DatasetItemService(ref.watch(dioClientProvider));
});

final itemFileServiceProvider = Provider<ItemFileService>((ref) {
  return ItemFileService(ref.watch(dioClientProvider));
});

// ==================== 图片输入历史服务 Provider ====================

final imageInputServiceProvider = Provider<ImageInputService>((ref) {
  return ImageInputService(ref.watch(dioClientProvider));
});

// ==================== 导入导出服务 Provider ====================

final importExportServiceProvider = Provider<ImportExportService>((ref) {
  return ImportExportService(ref.watch(dioClientProvider));
});

// ==================== API 密钥服务 Provider ====================

final apiKeyServiceProvider = Provider<ApiKeyService>((ref) {
  return ApiKeyService(ref.watch(dioClientProvider));
});
