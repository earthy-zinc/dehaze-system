import 'package:json_annotation/json_annotation.dart';

part 'notification_settings_model.g.dart';

/// 通知偏好设置
///
/// 契约字段固定为 5 个（pushEnabled / dndEnabled / dndStart / dndEnd / preferences），
/// GET 响应与 PATCH 请求体同构，故共用一个模型，不再单独定义 Form。
@JsonSerializable()
class NotificationSettings {
  const NotificationSettings({
    required this.pushEnabled,
    required this.dndEnabled,
    required this.preferences,
    this.dndStart,
    this.dndEnd,
  });

  factory NotificationSettings.fromJson(Map<String, dynamic> json) =>
      _$NotificationSettingsFromJson(json);

  /// APP 推送总开关
  final bool pushEnabled;

  /// 免打扰开关
  final bool dndEnabled;

  /// 免打扰开始时间（HH:mm:ss）
  final String? dndStart;

  /// 免打扰结束时间（HH:mm:ss）
  final String? dndEnd;

  final NotificationPreferences preferences;

  NotificationSettings copyWith({
    bool? pushEnabled,
    bool? dndEnabled,
    String? dndStart,
    String? dndEnd,
    NotificationPreferences? preferences,
  }) {
    return NotificationSettings(
      pushEnabled: pushEnabled ?? this.pushEnabled,
      dndEnabled: dndEnabled ?? this.dndEnabled,
      dndStart: dndStart ?? this.dndStart,
      dndEnd: dndEnd ?? this.dndEnd,
      preferences: preferences ?? this.preferences,
    );
  }

  Map<String, dynamic> toJson() => _$NotificationSettingsToJson(this);
}

/// 细粒度偏好设置
@JsonSerializable()
class NotificationPreferences {
  const NotificationPreferences({
    this.typeChannels = const {},
    this.moduleSwitches = const {},
  });

  factory NotificationPreferences.fromJson(Map<String, dynamic> json) =>
      _$NotificationPreferencesFromJson(json);

  /// 消息类型 → 推送渠道开关，key 取 announcement / business / member
  final Map<String, NotificationTypeChannel> typeChannels;

  /// 业务模块开关，key 取 prediction / feedback / announcement
  final Map<String, bool> moduleSwitches;

  /// 指定消息类型的推送开关，缺失时按开启处理（与新用户默认值一致）
  bool isTypeChannelPushEnabled(String type) =>
      typeChannels[type]?.push ?? true;

  /// 指定业务模块是否开启通知，缺失时按开启处理
  bool isModuleEnabled(String module) => moduleSwitches[module] ?? true;

  NotificationPreferences withTypeChannel(String type, bool push) {
    return NotificationPreferences(
      typeChannels: {
        ...typeChannels,
        type: NotificationTypeChannel(push: push),
      },
      moduleSwitches: moduleSwitches,
    );
  }

  NotificationPreferences withModuleSwitch(String module, bool enabled) {
    return NotificationPreferences(
      typeChannels: typeChannels,
      moduleSwitches: {...moduleSwitches, module: enabled},
    );
  }

  Map<String, dynamic> toJson() => _$NotificationPreferencesToJson(this);
}

/// 单条消息类型的推送渠道开关
@JsonSerializable()
class NotificationTypeChannel {
  const NotificationTypeChannel({required this.push});

  factory NotificationTypeChannel.fromJson(Map<String, dynamic> json) =>
      _$NotificationTypeChannelFromJson(json);

  final bool push;

  Map<String, dynamic> toJson() => _$NotificationTypeChannelToJson(this);
}
