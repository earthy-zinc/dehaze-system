/// 格式化工具函数
class FormatUtils {
  const FormatUtils._();

  /// 将字节数格式化为带单位的可读字符串（统一保留 1 位小数）
  ///
  /// - < 1 KB → "N B"
  /// - < 1 MB → "N.N KB"
  /// - 否则    → "N.N MB"
  static String formatFileSize(int bytes) {
    if (bytes < 1024) return '$bytes B';
    if (bytes < 1024 * 1024) return '${(bytes / 1024).toStringAsFixed(1)} KB';
    return '${(bytes / (1024 * 1024)).toStringAsFixed(1)} MB';
  }
}
