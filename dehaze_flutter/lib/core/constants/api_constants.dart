class ApiConstants {
  const ApiConstants._();

  static const String auth = '/auth';
  static const String files = '/files';
  static const String datasets = '/datasets';
  static const String datasetItems = '/dataset-items';
  static const String algorithm = '/algorithms';
  static const String prediction = '/prediction';
  static const String evaluation = '/evaluation';
  static const String recommendations = '/recommendations';

  static const String authLogin = '$auth/login';
  static const String authRegister = '$auth/register';
  static const String authLogout = '$auth/logout';
  static const String authCaptcha = '$auth/captcha';
  static const String authMe = '$auth/me';

  static const String filesUpload = files;

  static const String predictionLogs = '$prediction/logs';
  static const String evaluationLogs = '$evaluation/logs';

  static const String recommendationsAnalyze = '$recommendations/analyze';
  static const String recommendationsAlgorithms = '$recommendations/algorithms';
}
