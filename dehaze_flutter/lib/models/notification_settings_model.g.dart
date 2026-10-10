// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'notification_settings_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

NotificationSettings _$NotificationSettingsFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('NotificationSettings', json, ($checkedConvert) {
  final val = NotificationSettings(
    pushEnabled: $checkedConvert('pushEnabled', (v) => v as bool),
    dndEnabled: $checkedConvert('dndEnabled', (v) => v as bool),
    preferences: $checkedConvert(
      'preferences',
      (v) => NotificationPreferences.fromJson(v as Map<String, dynamic>),
    ),
    dndStart: $checkedConvert('dndStart', (v) => v as String?),
    dndEnd: $checkedConvert('dndEnd', (v) => v as String?),
  );
  return val;
});

Map<String, dynamic> _$NotificationSettingsToJson(
  NotificationSettings instance,
) => <String, dynamic>{
  'pushEnabled': instance.pushEnabled,
  'dndEnabled': instance.dndEnabled,
  'dndStart': ?instance.dndStart,
  'dndEnd': ?instance.dndEnd,
  'preferences': instance.preferences.toJson(),
};

NotificationPreferences _$NotificationPreferencesFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('NotificationPreferences', json, ($checkedConvert) {
  final val = NotificationPreferences(
    typeChannels: $checkedConvert(
      'typeChannels',
      (v) =>
          (v as Map<String, dynamic>?)?.map(
            (k, e) => MapEntry(
              k,
              NotificationTypeChannel.fromJson(e as Map<String, dynamic>),
            ),
          ) ??
          const {},
    ),
    moduleSwitches: $checkedConvert(
      'moduleSwitches',
      (v) =>
          (v as Map<String, dynamic>?)?.map((k, e) => MapEntry(k, e as bool)) ??
          const {},
    ),
  );
  return val;
});

Map<String, dynamic> _$NotificationPreferencesToJson(
  NotificationPreferences instance,
) => <String, dynamic>{
  'typeChannels': instance.typeChannels.map((k, e) => MapEntry(k, e.toJson())),
  'moduleSwitches': instance.moduleSwitches,
};

NotificationTypeChannel _$NotificationTypeChannelFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('NotificationTypeChannel', json, ($checkedConvert) {
  final val = NotificationTypeChannel(
    push: $checkedConvert('push', (v) => v as bool),
  );
  return val;
});

Map<String, dynamic> _$NotificationTypeChannelToJson(
  NotificationTypeChannel instance,
) => <String, dynamic>{'push': instance.push};
