class ApiEndpoints {
  // 基础路径
  static const String apiVersion = '/api/v1';

  // 认证相关
  static const String login = '$apiVersion/auth/login';
  static const String register = '$apiVersion/auth/register';
  static const String logout = '$apiVersion/auth/logout';
  static const String refreshToken = '$apiVersion/auth/refresh';
  static const String userProfile = '$apiVersion/auth/profile';

  // 算法管理
  static const String algorithms = '$apiVersion/algorithms';
  static const String algorithmDetail = '$apiVersion/algorithms/{id}';
  static const String recommendedAlgorithms =
      '$apiVersion/algorithms/recommend';
  static const String algorithmPerformance =
      '$apiVersion/algorithms/{id}/performance';
  static const String favoriteAlgorithms = '$apiVersion/algorithms/favorites';
  static const String toggleFavorite = '$apiVersion/algorithms/{id}/favorite';

  // 图像处理
  static const String imageProcessing = '$apiVersion/processing';
  static const String startProcessing = '$apiVersion/processing/start';
  static const String processingStatus =
      '$apiVersion/processing/{taskId}/status';
  static const String processingResult =
      '$apiVersion/processing/{taskId}/result';
  static const String pauseProcessing = '$apiVersion/processing/{taskId}/pause';
  static const String resumeProcessing =
      '$apiVersion/processing/{taskId}/resume';
  static const String cancelProcessing = '$apiVersion/processing/{taskId}';

  // 文件管理
  static const String fileUpload = '$apiVersion/files/upload';
  static const String batchUpload = '$apiVersion/files/batch-upload';
  static const String fileDownload = '$apiVersion/files/{fileId}/download';
  static const String filePreview = '$apiVersion/files/{fileId}/preview';
  static const String fileDelete = '$apiVersion/files/{fileId}';
  static const String fileList = '$apiVersion/files';

  // 历史记录
  static const String processingHistory = '$apiVersion/history';
  static const String historyDetail = '$apiVersion/history/{id}';
  static const String deleteHistory = '$apiVersion/history/{id}';
  static const String clearHistory = '$apiVersion/history/clear';

  // 用户设置
  static const String userSettings = '$apiVersion/settings';
  static const String updateSettings = '$apiVersion/settings/update';
  static const String resetSettings = '$apiVersion/settings/reset';

  // 统计和分析
  static const String usageStatistics = '$apiVersion/statistics/usage';
  static const String performanceMetrics = '$apiVersion/statistics/performance';
  static const String algorithmUsage = '$apiVersion/statistics/algorithms';

  // 系统信息
  static const String systemStatus = '$apiVersion/system/status';
  static const String systemInfo = '$apiVersion/system/info';
  static const String versionInfo = '$apiVersion/system/version';

  // WebSocket 连接
  static const String wsProcessingStatus = '/ws/processing';
  static const String wsNotifications = '/ws/notifications';
  static const String wsSystemStatus = '/ws/system';

  // 示例图像和样例数据
  static const String sampleImages = '$apiVersion/samples/images';
  static const String sampleResults = '$apiVersion/samples/results';

  // 帮助和反馈
  static const String help = '$apiVersion/help';
  static const String feedback = '$apiVersion/feedback';
  static const String reportIssue = '$apiVersion/feedback/issue';

  // 动态方法来构建带参数的URL
  static String algorithmDetailUrl(String id) =>
      algorithmDetail.replaceFirst('{id}', id);
  static String algorithmPerformanceUrl(String id) =>
      algorithmPerformance.replaceFirst('{id}', id);
  static String toggleFavoriteUrl(String id) =>
      toggleFavorite.replaceFirst('{id}', id);
  static String processingStatusUrl(String taskId) =>
      processingStatus.replaceFirst('{taskId}', taskId);
  static String processingResultUrl(String taskId) =>
      processingResult.replaceFirst('{taskId}', taskId);
  static String pauseProcessingUrl(String taskId) =>
      pauseProcessing.replaceFirst('{taskId}', taskId);
  static String resumeProcessingUrl(String taskId) =>
      resumeProcessing.replaceFirst('{taskId}', taskId);
  static String cancelProcessingUrl(String taskId) =>
      cancelProcessing.replaceFirst('{taskId}', taskId);
  static String fileDownloadUrl(String fileId) =>
      fileDownload.replaceFirst('{fileId}', fileId);
  static String filePreviewUrl(String fileId) =>
      filePreview.replaceFirst('{fileId}', fileId);
  static String fileDeleteUrl(String fileId) =>
      fileDelete.replaceFirst('{fileId}', fileId);
  static String historyDetailUrl(String id) =>
      historyDetail.replaceFirst('{id}', id);
  static String deleteHistoryUrl(String id) =>
      deleteHistory.replaceFirst('{id}', id);
}
