// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'notification_settings_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

NotificationSettings _$NotificationSettingsFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('NotificationSettings', json, ($checkedConvert) {
  final val = NotificationSettings(
    id: $checkedConvert('id', (v) => (v as num?)?.toInt()),
    userId: $checkedConvert('userId', (v) => (v as num?)?.toInt()),
    systemEnabled: $checkedConvert('systemEnabled', (v) => v as bool),
    predictionEnabled: $checkedConvert('predictionEnabled', (v) => v as bool),
    activityEnabled: $checkedConvert('activityEnabled', (v) => v as bool),
    announcementEnabled: $checkedConvert(
      'announcementEnabled',
      (v) => v as bool,
    ),
    emailEnabled: $checkedConvert('emailEnabled', (v) => v as bool),
    smsEnabled: $checkedConvert('smsEnabled', (v) => v as bool),
    pushEnabled: $checkedConvert('pushEnabled', (v) => v as bool),
    digestEnabled: $checkedConvert('digestEnabled', (v) => v as bool),
    digestFrequency: $checkedConvert('digestFrequency', (v) => v as String?),
    quietHoursEnabled: $checkedConvert('quietHoursEnabled', (v) => v as bool),
    quietStart: $checkedConvert('quietStart', (v) => v as String?),
    quietEnd: $checkedConvert('quietEnd', (v) => v as String?),
    createTime: $checkedConvert('createTime', (v) => v as String?),
    updateTime: $checkedConvert('updateTime', (v) => v as String?),
  );
  return val;
});

Map<String, dynamic> _$NotificationSettingsToJson(
  NotificationSettings instance,
) => <String, dynamic>{
  if (instance.id case final value?) 'id': value,
  if (instance.userId case final value?) 'userId': value,
  'systemEnabled': instance.systemEnabled,
  'predictionEnabled': instance.predictionEnabled,
  'activityEnabled': instance.activityEnabled,
  'announcementEnabled': instance.announcementEnabled,
  'emailEnabled': instance.emailEnabled,
  'smsEnabled': instance.smsEnabled,
  'pushEnabled': instance.pushEnabled,
  'digestEnabled': instance.digestEnabled,
  if (instance.digestFrequency case final value?) 'digestFrequency': value,
  'quietHoursEnabled': instance.quietHoursEnabled,
  if (instance.quietStart case final value?) 'quietStart': value,
  if (instance.quietEnd case final value?) 'quietEnd': value,
  if (instance.createTime case final value?) 'createTime': value,
  if (instance.updateTime case final value?) 'updateTime': value,
};

NotificationSettingsForm _$NotificationSettingsFormFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('NotificationSettingsForm', json, ($checkedConvert) {
  final val = NotificationSettingsForm(
    systemEnabled: $checkedConvert('systemEnabled', (v) => v as bool),
    predictionEnabled: $checkedConvert('predictionEnabled', (v) => v as bool),
    activityEnabled: $checkedConvert('activityEnabled', (v) => v as bool),
    announcementEnabled: $checkedConvert(
      'announcementEnabled',
      (v) => v as bool,
    ),
    emailEnabled: $checkedConvert('emailEnabled', (v) => v as bool),
    smsEnabled: $checkedConvert('smsEnabled', (v) => v as bool),
    pushEnabled: $checkedConvert('pushEnabled', (v) => v as bool),
    digestEnabled: $checkedConvert('digestEnabled', (v) => v as bool),
    digestFrequency: $checkedConvert('digestFrequency', (v) => v as String?),
    quietHoursEnabled: $checkedConvert('quietHoursEnabled', (v) => v as bool),
    quietStart: $checkedConvert('quietStart', (v) => v as String?),
    quietEnd: $checkedConvert('quietEnd', (v) => v as String?),
  );
  return val;
});

Map<String, dynamic> _$NotificationSettingsFormToJson(
  NotificationSettingsForm instance,
) => <String, dynamic>{
  'systemEnabled': instance.systemEnabled,
  'predictionEnabled': instance.predictionEnabled,
  'activityEnabled': instance.activityEnabled,
  'announcementEnabled': instance.announcementEnabled,
  'emailEnabled': instance.emailEnabled,
  'smsEnabled': instance.smsEnabled,
  'pushEnabled': instance.pushEnabled,
  'digestEnabled': instance.digestEnabled,
  if (instance.digestFrequency case final value?) 'digestFrequency': value,
  'quietHoursEnabled': instance.quietHoursEnabled,
  if (instance.quietStart case final value?) 'quietStart': value,
  if (instance.quietEnd case final value?) 'quietEnd': value,
};
