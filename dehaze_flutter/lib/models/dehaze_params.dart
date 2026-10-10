/// 去雾处理通用参数
///
/// 对齐产品文档 §2.3 通用模型参数：
/// - strength 去雾强度（0-100，默认 50）
/// - saturation 饱和度（0-200，默认 100）
/// - contrast 对比度（0-200，默认 100）
/// - sharpen 锐化（0-100，默认 30）
///
/// 强类型替代 `Map<String, dynamic>`，避免字符串 key 拼写错误无法在编译期发现。
/// [toJson] 仅包含与默认值不同的字段，避免干扰模型默认行为。
class DehazeParams {
  const DehazeParams({
    this.strength = defaultStrength,
    this.saturation = defaultSaturation,
    this.contrast = defaultContrast,
    this.sharpen = defaultSharpen,
  });

  /// 默认参数值（构造默认值、[isDefault]、[toJson] 与 UI 初值共用同一来源）
  static const int defaultStrength = 50;
  static const int defaultSaturation = 100;
  static const int defaultContrast = 100;
  static const int defaultSharpen = 30;

  /// 去雾强度 0-100
  final int strength;

  /// 饱和度 0-200
  final int saturation;

  /// 对比度 0-200
  final int contrast;

  /// 锐化 0-100
  final int sharpen;

  /// 是否全部为默认值（此时应传 null，不干扰模型默认行为）
  bool get isDefault =>
      strength == defaultStrength &&
      saturation == defaultSaturation &&
      contrast == defaultContrast &&
      sharpen == defaultSharpen;

  /// 序列化为请求参数 Map，仅包含与默认值不同的字段
  Map<String, dynamic> toJson() {
    final result = <String, dynamic>{};
    if (strength != defaultStrength) result['strength'] = strength;
    if (saturation != defaultSaturation) result['saturation'] = saturation;
    if (contrast != defaultContrast) result['contrast'] = contrast;
    if (sharpen != defaultSharpen) result['sharpen'] = sharpen;
    return result;
  }
}
