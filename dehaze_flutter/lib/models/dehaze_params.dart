/// 去雾处理通用参数
///
/// 对齐产品文档 §2.3 通用算法参数：
/// - strength 去雾强度（0-100，默认 50）
/// - saturation 饱和度（0-200，默认 100）
/// - contrast 对比度（0-200，默认 100）
/// - sharpen 锐化（0-100，默认 30）
///
/// 强类型替代 `Map<String, dynamic>`，避免字符串 key 拼写错误无法在编译期发现。
/// [toJson] 仅包含与默认值不同的字段，避免干扰算法默认行为。
class DehazeParams {
  const DehazeParams({
    this.strength = 50,
    this.saturation = 100,
    this.contrast = 100,
    this.sharpen = 30,
  });

  /// 去雾强度 0-100
  final int strength;

  /// 饱和度 0-200
  final int saturation;

  /// 对比度 0-200
  final int contrast;

  /// 锐化 0-100
  final int sharpen;

  /// 默认参数实例
  static const DehazeParams defaultParams = DehazeParams();

  /// 是否全部为默认值（此时应传 null，不干扰算法默认行为）
  bool get isDefault =>
      strength == 50 && saturation == 100 && contrast == 100 && sharpen == 30;

  /// 序列化为请求参数 Map，仅包含与默认值不同的字段
  Map<String, dynamic> toJson() {
    final result = <String, dynamic>{};
    if (strength != 50) result['strength'] = strength;
    if (saturation != 100) result['saturation'] = saturation;
    if (contrast != 100) result['contrast'] = contrast;
    if (sharpen != 30) result['sharpen'] = sharpen;
    return result;
  }
}
