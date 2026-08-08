// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'algorithm_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

AlgorithmModel _$AlgorithmModelFromJson(Map<String, dynamic> json) =>
    $checkedCreate('AlgorithmModel', json, ($checkedConvert) {
      final val = AlgorithmModel(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        name: $checkedConvert('name', (v) => v as String),
        type: $checkedConvert('type', (v) => v as String? ?? '未分类'),
        status: $checkedConvert(
          'status',
          (v) =>
              $enumDecodeNullable(_$AlgorithmStatusEnumMap, v) ??
              AlgorithmStatus.draft,
        ),
        parentId: $checkedConvert('parentId', (v) => (v as num?)?.toInt()),
        description: $checkedConvert('description', (v) => v as String?),
        img: $checkedConvert('img', (v) => v as String?),
        path: $checkedConvert('path', (v) => v as String?),
        importPath: $checkedConvert('importPath', (v) => v as String?),
        params: $checkedConvert('params', (v) => v as String?),
        flops: $checkedConvert('flops', (v) => v as String?),
        size: $checkedConvert('size', (v) => v as String?),
        version: $checkedConvert('version', (v) => v as String?),
        auditBy: $checkedConvert('auditBy', (v) => (v as num?)?.toInt()),
        auditTime: $checkedConvert('auditTime', (v) => v as String?),
        auditRemark: $checkedConvert('auditRemark', (v) => v as String?),
        createTime: $checkedConvert('createTime', (v) => v as String?),
        children: $checkedConvert(
          'children',
          (v) =>
              (v as List<dynamic>?)
                  ?.map(
                    (e) => AlgorithmModel.fromJson(e as Map<String, dynamic>),
                  )
                  .toList() ??
              const [],
        ),
      );
      return val;
    });

Map<String, dynamic> _$AlgorithmModelToJson(AlgorithmModel instance) =>
    <String, dynamic>{
      'id': instance.id,
      'name': instance.name,
      'type': instance.type,
      'status': _$AlgorithmStatusEnumMap[instance.status]!,
      if (instance.parentId case final value?) 'parentId': value,
      if (instance.description case final value?) 'description': value,
      if (instance.img case final value?) 'img': value,
      if (instance.path case final value?) 'path': value,
      if (instance.importPath case final value?) 'importPath': value,
      if (instance.params case final value?) 'params': value,
      if (instance.flops case final value?) 'flops': value,
      if (instance.size case final value?) 'size': value,
      if (instance.version case final value?) 'version': value,
      if (instance.auditBy case final value?) 'auditBy': value,
      if (instance.auditTime case final value?) 'auditTime': value,
      if (instance.auditRemark case final value?) 'auditRemark': value,
      if (instance.createTime case final value?) 'createTime': value,
      'children': instance.children.map((e) => e.toJson()).toList(),
    };

const _$AlgorithmStatusEnumMap = {
  AlgorithmStatus.draft: 0,
  AlgorithmStatus.testing: 1,
  AlgorithmStatus.pendingAudit: 2,
  AlgorithmStatus.published: 3,
  AlgorithmStatus.disabled: 4,
  AlgorithmStatus.archived: 5,
};

AlgorithmAuditForm _$AlgorithmAuditFormFromJson(Map<String, dynamic> json) =>
    $checkedCreate('AlgorithmAuditForm', json, ($checkedConvert) {
      final val = AlgorithmAuditForm(
        approved: $checkedConvert('approved', (v) => v as bool),
        remark: $checkedConvert('remark', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$AlgorithmAuditFormToJson(AlgorithmAuditForm instance) =>
    <String, dynamic>{
      'approved': instance.approved,
      if (instance.remark case final value?) 'remark': value,
    };

AlgorithmVersionForm _$AlgorithmVersionFormFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('AlgorithmVersionForm', json, ($checkedConvert) {
  final val = AlgorithmVersionForm(
    version: $checkedConvert('version', (v) => v as String),
    changeLog: $checkedConvert('changeLog', (v) => v as String?),
    modelFileId: $checkedConvert('modelFileId', (v) => (v as num?)?.toInt()),
  );
  return val;
});

Map<String, dynamic> _$AlgorithmVersionFormToJson(
  AlgorithmVersionForm instance,
) => <String, dynamic>{
  'version': instance.version,
  if (instance.changeLog case final value?) 'changeLog': value,
  if (instance.modelFileId case final value?) 'modelFileId': value,
};

AlgorithmVersionVO _$AlgorithmVersionVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('AlgorithmVersionVO', json, ($checkedConvert) {
      final val = AlgorithmVersionVO(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        algorithmId: $checkedConvert('algorithmId', (v) => (v as num).toInt()),
        version: $checkedConvert('version', (v) => v as String),
        changeLog: $checkedConvert('changeLog', (v) => v as String?),
        status: $checkedConvert('status', (v) => (v as num?)?.toInt()),
        isActive: $checkedConvert('isActive', (v) => v as bool?),
        modelFileId: $checkedConvert(
          'modelFileId',
          (v) => (v as num?)?.toInt(),
        ),
        createTime: $checkedConvert('createTime', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$AlgorithmVersionVOToJson(AlgorithmVersionVO instance) =>
    <String, dynamic>{
      'id': instance.id,
      'algorithmId': instance.algorithmId,
      'version': instance.version,
      if (instance.changeLog case final value?) 'changeLog': value,
      if (instance.status case final value?) 'status': value,
      if (instance.isActive case final value?) 'isActive': value,
      if (instance.modelFileId case final value?) 'modelFileId': value,
      if (instance.createTime case final value?) 'createTime': value,
    };

AlgorithmMonitorVO _$AlgorithmMonitorVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('AlgorithmMonitorVO', json, ($checkedConvert) {
      final val = AlgorithmMonitorVO(
        callCount: $checkedConvert('callCount', (v) => (v as num).toInt()),
        avgTime: $checkedConvert('avgTime', (v) => (v as num).toDouble()),
        successRate: $checkedConvert(
          'successRate',
          (v) => (v as num).toDouble(),
        ),
        todayCallCount: $checkedConvert(
          'todayCallCount',
          (v) => (v as num).toInt(),
        ),
      );
      return val;
    });

Map<String, dynamic> _$AlgorithmMonitorVOToJson(AlgorithmMonitorVO instance) =>
    <String, dynamic>{
      'callCount': instance.callCount,
      'avgTime': instance.avgTime,
      'successRate': instance.successRate,
      'todayCallCount': instance.todayCallCount,
    };

AlgorithmMonitorStatsItemVO _$AlgorithmMonitorStatsItemVOFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('AlgorithmMonitorStatsItemVO', json, ($checkedConvert) {
  final val = AlgorithmMonitorStatsItemVO(
    date: $checkedConvert('date', (v) => v as String),
    callCount: $checkedConvert('callCount', (v) => (v as num).toInt()),
    avgTime: $checkedConvert('avgTime', (v) => (v as num).toDouble()),
    successRate: $checkedConvert('successRate', (v) => (v as num).toDouble()),
  );
  return val;
});

Map<String, dynamic> _$AlgorithmMonitorStatsItemVOToJson(
  AlgorithmMonitorStatsItemVO instance,
) => <String, dynamic>{
  'date': instance.date,
  'callCount': instance.callCount,
  'avgTime': instance.avgTime,
  'successRate': instance.successRate,
};

AlgorithmCompareVO _$AlgorithmCompareVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('AlgorithmCompareVO', json, ($checkedConvert) {
      final val = AlgorithmCompareVO(
        algorithmId: $checkedConvert('algorithmId', (v) => (v as num).toInt()),
        algorithmName: $checkedConvert('algorithmName', (v) => v as String),
        resultUrl: $checkedConvert('resultUrl', (v) => v as String?),
        time: $checkedConvert('time', (v) => (v as num?)?.toInt()),
        metrics: $checkedConvert('metrics', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$AlgorithmCompareVOToJson(AlgorithmCompareVO instance) =>
    <String, dynamic>{
      'algorithmId': instance.algorithmId,
      'algorithmName': instance.algorithmName,
      if (instance.resultUrl case final value?) 'resultUrl': value,
      if (instance.time case final value?) 'time': value,
      if (instance.metrics case final value?) 'metrics': value,
    };

AlgorithmSelectNodeVO _$AlgorithmSelectNodeVOFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('AlgorithmSelectNodeVO', json, ($checkedConvert) {
  final val = AlgorithmSelectNodeVO(
    id: $checkedConvert('id', (v) => (v as num).toInt()),
    parentId: $checkedConvert('parentId', (v) => (v as num).toInt()),
    name: $checkedConvert('name', (v) => v as String),
    type: $checkedConvert('type', (v) => v as String),
    leaf: $checkedConvert('leaf', (v) => v as bool),
    children: $checkedConvert(
      'children',
      (v) => (v as List<dynamic>?)
          ?.map(
            (e) => AlgorithmSelectNodeVO.fromJson(e as Map<String, dynamic>),
          )
          .toList(),
    ),
  );
  return val;
});

Map<String, dynamic> _$AlgorithmSelectNodeVOToJson(
  AlgorithmSelectNodeVO instance,
) => <String, dynamic>{
  'id': instance.id,
  'parentId': instance.parentId,
  'name': instance.name,
  'type': instance.type,
  'leaf': instance.leaf,
  if (instance.children?.map((e) => e.toJson()).toList() case final value?)
    'children': value,
};

AlgorithmDetailVO _$AlgorithmDetailVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('AlgorithmDetailVO', json, ($checkedConvert) {
      final val = AlgorithmDetailVO(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        name: $checkedConvert('name', (v) => v as String),
        type: $checkedConvert('type', (v) => v as String),
        description: $checkedConvert('description', (v) => v as String),
        img: $checkedConvert('img', (v) => v as String?),
        path: $checkedConvert('path', (v) => v as String?),
        size: $checkedConvert('size', (v) => v as String?),
        params: $checkedConvert('params', (v) => v as String?),
        flops: $checkedConvert('flops', (v) => v as String?),
        version: $checkedConvert('version', (v) => v as String?),
        status: $checkedConvert('status', (v) => (v as num?)?.toInt()),
        avgRating: $checkedConvert('avgRating', (v) => (v as num?)?.toDouble()),
        ratingCount: $checkedConvert(
          'ratingCount',
          (v) => (v as num?)?.toInt(),
        ),
        usageCount: $checkedConvert('usageCount', (v) => (v as num?)?.toInt()),
        sampleImages: $checkedConvert(
          'sampleImages',
          (v) => (v as List<dynamic>?)?.map((e) => e as String).toList(),
        ),
      );
      return val;
    });

Map<String, dynamic> _$AlgorithmDetailVOToJson(AlgorithmDetailVO instance) =>
    <String, dynamic>{
      'id': instance.id,
      'name': instance.name,
      'type': instance.type,
      if (instance.img case final value?) 'img': value,
      'description': instance.description,
      if (instance.path case final value?) 'path': value,
      if (instance.size case final value?) 'size': value,
      if (instance.params case final value?) 'params': value,
      if (instance.flops case final value?) 'flops': value,
      if (instance.version case final value?) 'version': value,
      if (instance.status case final value?) 'status': value,
      if (instance.avgRating case final value?) 'avgRating': value,
      if (instance.ratingCount case final value?) 'ratingCount': value,
      if (instance.usageCount case final value?) 'usageCount': value,
      if (instance.sampleImages case final value?) 'sampleImages': value,
    };
