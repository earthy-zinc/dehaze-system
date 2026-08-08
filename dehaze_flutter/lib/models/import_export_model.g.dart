// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'import_export_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

ImportRecordVO _$ImportRecordVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('ImportRecordVO', json, ($checkedConvert) {
      final val = ImportRecordVO(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        type: $checkedConvert('type', (v) => v as String),
        fileName: $checkedConvert('fileName', (v) => v as String?),
        fileSize: $checkedConvert('fileSize', (v) => (v as num?)?.toInt()),
        status: $checkedConvert('status', (v) => (v as num).toInt()),
        statusName: $checkedConvert('statusName', (v) => v as String?),
        total: $checkedConvert('total', (v) => (v as num?)?.toInt()),
        success: $checkedConvert('success', (v) => (v as num?)?.toInt()),
        failed: $checkedConvert('failed', (v) => (v as num?)?.toInt()),
        errorMessage: $checkedConvert('errorMessage', (v) => v as String?),
        createTime: $checkedConvert('createTime', (v) => v as String?),
        completeTime: $checkedConvert('completeTime', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$ImportRecordVOToJson(ImportRecordVO instance) =>
    <String, dynamic>{
      'id': instance.id,
      'type': instance.type,
      if (instance.fileName case final value?) 'fileName': value,
      if (instance.fileSize case final value?) 'fileSize': value,
      'status': instance.status,
      if (instance.statusName case final value?) 'statusName': value,
      if (instance.total case final value?) 'total': value,
      if (instance.success case final value?) 'success': value,
      if (instance.failed case final value?) 'failed': value,
      if (instance.errorMessage case final value?) 'errorMessage': value,
      if (instance.createTime case final value?) 'createTime': value,
      if (instance.completeTime case final value?) 'completeTime': value,
    };

ImportRecordQuery _$ImportRecordQueryFromJson(Map<String, dynamic> json) =>
    $checkedCreate('ImportRecordQuery', json, ($checkedConvert) {
      final val = ImportRecordQuery(
        pageNum: $checkedConvert('pageNum', (v) => (v as num?)?.toInt() ?? 1),
        pageSize: $checkedConvert(
          'pageSize',
          (v) => (v as num?)?.toInt() ?? 10,
        ),
        type: $checkedConvert('type', (v) => v as String?),
        status: $checkedConvert('status', (v) => (v as num?)?.toInt()),
        startDate: $checkedConvert('startDate', (v) => v as String?),
        endDate: $checkedConvert('endDate', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$ImportRecordQueryToJson(ImportRecordQuery instance) =>
    <String, dynamic>{
      'pageNum': instance.pageNum,
      'pageSize': instance.pageSize,
      if (instance.type case final value?) 'type': value,
      if (instance.status case final value?) 'status': value,
      if (instance.startDate case final value?) 'startDate': value,
      if (instance.endDate case final value?) 'endDate': value,
    };

ExportRecordVO _$ExportRecordVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('ExportRecordVO', json, ($checkedConvert) {
      final val = ExportRecordVO(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        type: $checkedConvert('type', (v) => v as String),
        fileName: $checkedConvert('fileName', (v) => v as String?),
        fileSize: $checkedConvert('fileSize', (v) => (v as num?)?.toInt()),
        status: $checkedConvert('status', (v) => (v as num).toInt()),
        statusName: $checkedConvert('statusName', (v) => v as String?),
        total: $checkedConvert('total', (v) => (v as num?)?.toInt()),
        downloadUrl: $checkedConvert('downloadUrl', (v) => v as String?),
        errorMessage: $checkedConvert('errorMessage', (v) => v as String?),
        createTime: $checkedConvert('createTime', (v) => v as String?),
        completeTime: $checkedConvert('completeTime', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$ExportRecordVOToJson(ExportRecordVO instance) =>
    <String, dynamic>{
      'id': instance.id,
      'type': instance.type,
      if (instance.fileName case final value?) 'fileName': value,
      if (instance.fileSize case final value?) 'fileSize': value,
      'status': instance.status,
      if (instance.statusName case final value?) 'statusName': value,
      if (instance.total case final value?) 'total': value,
      if (instance.downloadUrl case final value?) 'downloadUrl': value,
      if (instance.errorMessage case final value?) 'errorMessage': value,
      if (instance.createTime case final value?) 'createTime': value,
      if (instance.completeTime case final value?) 'completeTime': value,
    };

ExportRecordQuery _$ExportRecordQueryFromJson(Map<String, dynamic> json) =>
    $checkedCreate('ExportRecordQuery', json, ($checkedConvert) {
      final val = ExportRecordQuery(
        pageNum: $checkedConvert('pageNum', (v) => (v as num?)?.toInt() ?? 1),
        pageSize: $checkedConvert(
          'pageSize',
          (v) => (v as num?)?.toInt() ?? 10,
        ),
        type: $checkedConvert('type', (v) => v as String?),
        status: $checkedConvert('status', (v) => (v as num?)?.toInt()),
        startDate: $checkedConvert('startDate', (v) => v as String?),
        endDate: $checkedConvert('endDate', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$ExportRecordQueryToJson(ExportRecordQuery instance) =>
    <String, dynamic>{
      'pageNum': instance.pageNum,
      'pageSize': instance.pageSize,
      if (instance.type case final value?) 'type': value,
      if (instance.status case final value?) 'status': value,
      if (instance.startDate case final value?) 'startDate': value,
      if (instance.endDate case final value?) 'endDate': value,
    };

ImportTemplateVO _$ImportTemplateVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('ImportTemplateVO', json, ($checkedConvert) {
      final val = ImportTemplateVO(
        type: $checkedConvert('type', (v) => v as String),
        name: $checkedConvert('name', (v) => v as String),
        description: $checkedConvert('description', (v) => v as String?),
        url: $checkedConvert('url', (v) => v as String?),
        columns: $checkedConvert(
          'columns',
          (v) => (v as List<dynamic>).map((e) => e as String).toList(),
        ),
      );
      return val;
    });

Map<String, dynamic> _$ImportTemplateVOToJson(ImportTemplateVO instance) =>
    <String, dynamic>{
      'type': instance.type,
      'name': instance.name,
      if (instance.description case final value?) 'description': value,
      if (instance.url case final value?) 'url': value,
      'columns': instance.columns,
    };
