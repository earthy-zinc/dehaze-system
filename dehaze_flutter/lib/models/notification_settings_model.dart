import 'package:json_annotation/json_annotation.dart';

part 'notification_settings_model.g.dart';

// ==================== 通知设置 ====================

@JsonSerializable()
class NotificationSettings {
  const NotificationSettings({
    this.id,
    this.userId,
    required this.systemEnabled,
    required this.predictionEnabled,
    required this.activityEnabled,
    required this.announcementEnabled,
    required this.emailEnabled,
    required this.smsEnabled,
    required this.pushEnabled,
    required this.digestEnabled,
    this.digestFrequency,
    required this.quietHoursEnabled,
    this.quietStart,
    this.quietEnd,
    this.createTime,
    this.updateTime,
  });

  factory NotificationSettings.fromJson(Map<String, dynamic> json) =>
      _$NotificationSettingsFromJson(json);

  final int? id;
  final int? userId;

  @JsonKey(name: 'systemEnabled')
  final bool systemEnabled;

  @JsonKey(name: 'predictionEnabled')
  final bool predictionEnabled;

  @JsonKey(name: 'activityEnabled')
  final bool activityEnabled;

  @JsonKey(name: 'announcementEnabled')
  final bool announcementEnabled;

  @JsonKey(name: 'emailEnabled')
  final bool emailEnabled;

  @JsonKey(name: 'smsEnabled')
  final bool smsEnabled;

  @JsonKey(name: 'pushEnabled')
  final bool pushEnabled;

  @JsonKey(name: 'digestEnabled')
  final bool digestEnabled;

  @JsonKey(name: 'digestFrequency')
  final String? digestFrequency;

  @JsonKey(name: 'quietHoursEnabled')
  final bool quietHoursEnabled;

  @JsonKey(name: 'quietStart')
  final String? quietStart;

  @JsonKey(name: 'quietEnd')
  final String? quietEnd;

  final String? createTime;
  final String? updateTime;

  Map<String, dynamic> toJson() => _$NotificationSettingsToJson(this);
}

// ==================== 通知设置表单 ====================

@JsonSerializable()
class NotificationSettingsForm {
  const NotificationSettingsForm({
    required this.systemEnabled,
    required this.predictionEnabled,
    required this.activityEnabled,
    required this.announcementEnabled,
    required this.emailEnabled,
    required this.smsEnabled,
    required this.pushEnabled,
    required this.digestEnabled,
    this.digestFrequency,
    required this.quietHoursEnabled,
    this.quietStart,
    this.quietEnd,
  });

  factory NotificationSettingsForm.fromJson(Map<String, dynamic> json) =>
      _$NotificationSettingsFormFromJson(json);

  @JsonKey(name: 'systemEnabled')
  final bool systemEnabled;

  @JsonKey(name: 'predictionEnabled')
  final bool predictionEnabled;

  @JsonKey(name: 'activityEnabled')
  final bool activityEnabled;

  @JsonKey(name: 'announcementEnabled')
  final bool announcementEnabled;

  @JsonKey(name: 'emailEnabled')
  final bool emailEnabled;

  @JsonKey(name: 'smsEnabled')
  final bool smsEnabled;

  @JsonKey(name: 'pushEnabled')
  final bool pushEnabled;

  @JsonKey(name: 'digestEnabled')
  final bool digestEnabled;

  @JsonKey(name: 'digestFrequency')
  final String? digestFrequency;

  @JsonKey(name: 'quietHoursEnabled')
  final bool quietHoursEnabled;

  @JsonKey(name: 'quietStart')
  final String? quietStart;

  @JsonKey(name: 'quietEnd')
  final String? quietEnd;

  Map<String, dynamic> toJson() => _$NotificationSettingsFormToJson(this);
}
