class ApiConstants {
  const ApiConstants._();

  // Auth
  static const String auth = '/auth';
  static const String authLogin = '$auth/login';
  static const String authRegister = '$auth/register';
  static const String authLogout = '$auth/logout';
  static const String authCaptcha = '$auth/captcha';
  static const String authMe = '$auth/me';
  static const String authApiKeys = '$auth/api-keys';

  // Files
  static const String files = '/files';
  static const String filesCheck = '$files/check';
  static const String filesPage = '$files/page';
  static const String filesDownload = '$files/download';

  // Datasets
  static const String datasets = '/datasets';
  static const String datasetsChildren = '$datasets/children';
  static const String datasetsOptions = '$datasets/options';
  static const String datasetsBatch = '$datasets/batch';

  // Dataset Items
  static const String datasetItems = '/dataset-items';
  static const String datasetItemsUpload = '$datasetItems/upload';
  static const String datasetItemsBatch = '$datasetItems/batch';

  // Item Files
  static const String itemFiles = '/item-files';
  static const String itemFilesBatch = '$itemFiles/batch';

  // Algorithms
  static const String algorithm = '/algorithms';
  static const String algorithmOptions = '$algorithm/options';
  static const String algorithmList = '$algorithm/list';
  static const String algorithmSelect = '$algorithm/select';
  static const String algorithmSelectTree = '$algorithm/select/tree';
  static const String algorithmSelectCompare = '$algorithm/select/compare';
  static const String algorithmSelectSearch = '$algorithm/select/search';

  // Prediction
  static const String prediction = '/prediction';
  static const String predictionLogs = '$prediction/logs';
  static const String predictionBatch = '$prediction/batch';
  static const String predictionQuota = '$prediction/quota';

  // Evaluation
  static const String evaluation = '/evaluation';
  static const String evaluationLogs = '$evaluation/logs';
  static const String evaluationMetrics = '$evaluation/metrics';

  // Presets
  static const String presets = '/presets';

  // Compare
  static const String compare = '/compare';
  static const String compareReport = '$compare/report';

  // Recommendations
  static const String recommendations = '/recommendations';
  static const String recommendationsAnalyze = '$recommendations/analyze';
  static const String recommendationsAlgorithms = '$recommendations/algorithms';
  static const String recommendationsFeedback = '$recommendations/feedback';
  static const String recommendationsRules = '$recommendations/rules';
  static const String recommendationsReport = '$recommendations/report';

  // Users
  static const String users = '/users';
  static const String usersPage = '$users/page';

  // Roles
  static const String roles = '/roles';
  static const String rolesPage = '$roles/page';
  static const String rolesOptions = '$roles/options';

  // Menus
  static const String menus = '/menus';
  static const String menusRoutes = '$menus/routes';
  static const String menusOptions = '$menus/options';

  // Depts
  static const String depts = '/depts';
  static const String deptsOptions = '$depts/options';

  // Dict
  static const String dict = '/dict';
  static const String dictTypes = '$dict/types';
  static const String dictTypesPage = '$dictTypes/page';
  static const String dictPage = '$dict/page';

  // Tasks
  static const String tasks = '/tasks';

  // Members
  static const String members = '/members';
  static const String membersProfile = '$members/profile';
  static const String membersGrowthLogs = '$members/growth-logs';
  static const String membersSignIn = '$members/sign-in';
  static const String membersSignInCalendar = '$members/sign-in/calendar';
  static const String membersPage = '$members/page';
  static const String membersBenefits = '$members/benefits';

  // Packages
  static const String packages = '/packages';
  static const String packagesPage = '$packages/page';
  static const String packagesCalculatePrice = '$packages/calculate-price';
  static const String packagesCoupons = '$packages/coupons';
  static const String packagesCouponsPage = '$packages/coupons/page';
  static const String packagesCouponsMy = '$packages/coupons/my';
  static const String packagesCouponsBatch = '$packages/coupons/batch';
  static const String packagesSalesStats = '$packages/sales/stats';

  // Orders
  static const String orders = '/orders';
  static const String ordersMy = '$orders/my';
  static const String ordersPage = '$orders/page';
  static const String ordersRefunds = '$orders/refunds';
  static const String ordersRefundsPage = '$orders/refunds/page';
  static const String ordersAutoRenewConfig = '$orders/auto-renew/config';
  static const String ordersStats = '$orders/stats';

  // Feedback
  static const String feedback = '/feedback';
  static const String feedbackRatings = '$feedback/ratings';
  static const String feedbackRatingsMy = '$feedback/ratings/my';
  static const String feedbackRatingsPage = '$feedback/ratings/page';
  static const String feedbackRatingsStats = '$feedback/ratings/stats';
  static const String feedbackRatingsByPrediction = '$feedback/ratings/by-prediction';
  static const String feedbackMy = '$feedback/my';
  static const String feedbackPage = '$feedback/page';
  static const String feedbackStats = '$feedback/stats';

  // Favorites
  static const String favorites = '/favorites';
  static const String favoritesPage = '$favorites/page';
  static const String favoritesCount = '$favorites/count';

  // Messages
  static const String messages = '/messages';
  static const String messagesUnreadCount = '$messages/unread-count';
  static const String messagesReadAll = '$messages/_read-all';
  static const String messagesSearch = '$messages/search';
  static const String messagesSend = '$messages/send';

  // Announcements
  static const String announcements = '/announcements';
  static const String announcementsPage = '$announcements/page';

  // Message Templates
  static const String messageTemplates = '/message-templates';
  static const String messageTemplatesPage = '$messageTemplates/page';

  // Notification Settings
  static const String notificationSettings = '/notification-settings';

  // Image Input History
  static const String imageInputHistory = '/image-input/history';
  static const String imageInputHistoryBatch = '$imageInputHistory/batch';
  static const String imageInputHistoryClear = '$imageInputHistory/clear';

  // Import Export
  static const String importExport = '/import-export';
}
