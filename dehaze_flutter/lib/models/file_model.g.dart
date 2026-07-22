// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'file_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

FileUploadResponse _$FileUploadResponseFromJson(Map<String, dynamic> json) =>
    $checkedCreate('FileUploadResponse', json, ($checkedConvert) {
      final val = FileUploadResponse(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        url: $checkedConvert('url', (v) => v as String),
        name: $checkedConvert('name', (v) => v as String),
        type: $checkedConvert('type', (v) => v as String?),
        objectName: $checkedConvert('objectName', (v) => v as String?),
        size: $checkedConvert('size', (v) => v as String?),
        path: $checkedConvert('path', (v) => v as String?),
        md5: $checkedConvert('md5', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$FileUploadResponseToJson(FileUploadResponse instance) =>
    <String, dynamic>{
      'id': instance.id,
      'url': instance.url,
      'name': instance.name,
      if (instance.type case final value?) 'type': value,
      if (instance.objectName case final value?) 'objectName': value,
      if (instance.size case final value?) 'size': value,
      if (instance.path case final value?) 'path': value,
      if (instance.md5 case final value?) 'md5': value,
    };
