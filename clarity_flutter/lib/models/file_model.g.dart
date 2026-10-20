// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'file_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

FileInfo _$FileInfoFromJson(Map<String, dynamic> json) =>
    $checkedCreate('FileInfo', json, ($checkedConvert) {
      final val = FileInfo(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        url: $checkedConvert('url', (v) => v as String),
        name: $checkedConvert('name', (v) => v as String),
        type: $checkedConvert('type', (v) => v as String?),
        objectName: $checkedConvert('objectName', (v) => v as String?),
        size: $checkedConvert('size', (v) => v as String?),
        sizeBytes: $checkedConvert('sizeBytes', (v) => (v as num?)?.toInt()),
        storage: $checkedConvert('storage', (v) => v as String?),
        md5: $checkedConvert('md5', (v) => v as String?),
        createTime: $checkedConvert('createTime', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$FileInfoToJson(FileInfo instance) => <String, dynamic>{
  'id': instance.id,
  'url': instance.url,
  'name': instance.name,
  'type': ?instance.type,
  'objectName': ?instance.objectName,
  'size': ?instance.size,
  'sizeBytes': ?instance.sizeBytes,
  'storage': ?instance.storage,
  'md5': ?instance.md5,
  'createTime': ?instance.createTime,
};
