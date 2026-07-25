class ApiConstants {
  const ApiConstants._();

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

  static const String authLogin = '$auth/login';
  static const String authLogout = '$auth/logout';
  static const String authCaptcha = '$auth/captcha';
  static const String authMe = '$auth/me';

  static const String filesUpload = files;
  static const String filesDownload = '$files/download';
  static const String filesCheck = '$files/check';
  static const String filesPage = '$files/page';

  static const String datasetsOptions = '$datasets/options';
  static const String datasetsBatch = '$datasets/batch';
  static const String datasetItemsUpload = '$datasetItems/upload';
  static const String datasetItemsBatch = '$datasetItems/batch';

  static const String algorithmOptions = '$algorithm/options';
  static const String algorithmExport = '$algorithm/_export';
  static const String algorithmImport = '$algorithm/_import';

  static const String predictionLogs = '$prediction/logs';
  static const String evaluationLogs = '$evaluation/logs';

  static const String usersPage = '$users/page';

  static const String successCode = '00000';
}
