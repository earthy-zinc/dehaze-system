// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'file_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

FileModel _$FileModelFromJson(Map<String, dynamic> json) =>
    $checkedCreate('FileModel', json, ($checkedConvert) {
      final val = FileModel(
        fileId: $checkedConvert('fileId', (v) => v as String),
        fileName: $checkedConvert('fileName', (v) => v as String),
        fileUrl: $checkedConvert('fileUrl', (v) => v as String),
        fileSize: $checkedConvert('fileSize', (v) => (v as num).toInt()),
        fileType: $checkedConvert('fileType', (v) => v as String),
        md5: $checkedConvert('md5', (v) => v as String?),
        objectName: $checkedConvert('objectName', (v) => v as String?),
        createTime: $checkedConvert('createTime', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$FileModelToJson(FileModel instance) => <String, dynamic>{
  'fileId': instance.fileId,
  'fileName': instance.fileName,
  'fileUrl': instance.fileUrl,
  'fileSize': instance.fileSize,
  'fileType': instance.fileType,
  if (instance.md5 case final value?) 'md5': value,
  if (instance.objectName case final value?) 'objectName': value,
  if (instance.createTime case final value?) 'createTime': value,
};

FileUploadResponse _$FileUploadResponseFromJson(Map<String, dynamic> json) =>
    $checkedCreate('FileUploadResponse', json, ($checkedConvert) {
      final val = FileUploadResponse(
        fileId: $checkedConvert('fileId', (v) => v as String),
        fileUrl: $checkedConvert('fileUrl', (v) => v as String),
        fileName: $checkedConvert('fileName', (v) => v as String),
        fileSize: $checkedConvert('fileSize', (v) => (v as num).toInt()),
        md5: $checkedConvert('md5', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$FileUploadResponseToJson(FileUploadResponse instance) =>
    <String, dynamic>{
      'fileId': instance.fileId,
      'fileUrl': instance.fileUrl,
      'fileName': instance.fileName,
      'fileSize': instance.fileSize,
      if (instance.md5 case final value?) 'md5': value,
    };
