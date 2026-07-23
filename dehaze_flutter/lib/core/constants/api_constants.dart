/// API 路径常量
///
/// 统一管理所有后端 API 路径，与后端接口清单保持一致
/// 参考：02-系统架构/04-API规范.md
class ApiConstants {
  const ApiConstants._();

  // ==================== 基础路径 ====================
  static const String auth = '/auth';
  static const String users = '/users';
  static const String roles = '/roles';
  static const String menus = '/menus';
  static const String dept = '/dept';
  static const String dict = '/dict';
  static const String files = '/files';
  static const String datasets = '/datasets';
  static const String datasetItems = '/dataset-items';
  static const String itemFiles = '/item-files';
  static const String algorithm = '/algorithms';
  static const String prediction = '/prediction';
  static const String evaluation = '/evaluation';
  static const String tasks = '/tasks';

  // ==================== 认证管理 ====================
  static const String authLogin = '$auth/login';
  static const String authLogout = '$auth/logout';
  static const String authCaptcha = '$auth/captcha';
  static const String authRefresh = '$auth/refresh';
  static const String authMe = '$auth/me';

  // ==================== 文件管理 ====================
  static const String filesUpload = files;
  static const String filesDownload = '$files/download';
  static const String filesCheck = '$files/check';
  static const String filesPage = '$files/page';

  // ==================== 数据集管理 ====================
  static const String datasetsOptions = '$datasets/options';
  static const String datasetsBatch = '$datasets/batch';
  static const String datasetItemsUpload = '$datasetItems/upload';
  static const String datasetItemsBatch = '$datasetItems/batch';

  // ==================== 算法管理 ====================
  static const String algorithmOptions = '$algorithm/options';
  static const String algorithmExport = '$algorithm/_export';
  static const String algorithmImport = '$algorithm/_import';

  // ==================== 预测/评估 ====================
  static const String predictionLogs = '$prediction/logs';
  static const String evaluationLogs = '$evaluation/logs';

  // ==================== 用户管理 ====================
  static const String usersPage = '$users/page';

  // ==================== 成功状态码 ====================
  static const String successCode = '00000';
}
