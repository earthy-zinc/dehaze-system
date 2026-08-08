// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'dict_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

DictType _$DictTypeFromJson(Map<String, dynamic> json) =>
    $checkedCreate('DictType', json, ($checkedConvert) {
      final val = DictType(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        name: $checkedConvert('name', (v) => v as String),
        code: $checkedConvert('code', (v) => v as String),
        status: $checkedConvert('status', (v) => (v as num?)?.toInt()),
        remark: $checkedConvert('remark', (v) => v as String?),
        createTime: $checkedConvert('createTime', (v) => v as String?),
        updateTime: $checkedConvert('updateTime', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$DictTypeToJson(DictType instance) => <String, dynamic>{
  'id': instance.id,
  'name': instance.name,
  'code': instance.code,
  if (instance.status case final value?) 'status': value,
  if (instance.remark case final value?) 'remark': value,
  if (instance.createTime case final value?) 'createTime': value,
  if (instance.updateTime case final value?) 'updateTime': value,
};

DictTypeQuery _$DictTypeQueryFromJson(Map<String, dynamic> json) =>
    $checkedCreate('DictTypeQuery', json, ($checkedConvert) {
      final val = DictTypeQuery(
        pageNum: $checkedConvert('pageNum', (v) => (v as num?)?.toInt()),
        pageSize: $checkedConvert('pageSize', (v) => (v as num?)?.toInt()),
        keywords: $checkedConvert('keywords', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$DictTypeQueryToJson(DictTypeQuery instance) =>
    <String, dynamic>{
      if (instance.pageNum case final value?) 'pageNum': value,
      if (instance.pageSize case final value?) 'pageSize': value,
      if (instance.keywords case final value?) 'keywords': value,
    };

DictTypeForm _$DictTypeFormFromJson(Map<String, dynamic> json) =>
    $checkedCreate('DictTypeForm', json, ($checkedConvert) {
      final val = DictTypeForm(
        id: $checkedConvert('id', (v) => (v as num?)?.toInt()),
        name: $checkedConvert('name', (v) => v as String?),
        code: $checkedConvert('code', (v) => v as String?),
        status: $checkedConvert('status', (v) => (v as num).toInt()),
        remark: $checkedConvert('remark', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$DictTypeFormToJson(DictTypeForm instance) =>
    <String, dynamic>{
      if (instance.id case final value?) 'id': value,
      if (instance.name case final value?) 'name': value,
      if (instance.code case final value?) 'code': value,
      'status': instance.status,
      if (instance.remark case final value?) 'remark': value,
    };

Dict _$DictFromJson(Map<String, dynamic> json) =>
    $checkedCreate('Dict', json, ($checkedConvert) {
      final val = Dict(
        id: $checkedConvert('id', (v) => (v as num?)?.toInt()),
        typeId: $checkedConvert('typeId', (v) => (v as num?)?.toInt()),
        typeCode: $checkedConvert('typeCode', (v) => v as String?),
        label: $checkedConvert('label', (v) => v as String?),
        value: $checkedConvert('value', (v) => v as String?),
        sort: $checkedConvert('sort', (v) => (v as num?)?.toInt()),
        status: $checkedConvert('status', (v) => (v as num?)?.toInt()),
        remark: $checkedConvert('remark', (v) => v as String?),
        cssClass: $checkedConvert('cssClass', (v) => v as String?),
        listClass: $checkedConvert('listClass', (v) => v as String?),
        isDefault: $checkedConvert('isDefault', (v) => (v as num?)?.toInt()),
        createTime: $checkedConvert('createTime', (v) => v as String?),
        updateTime: $checkedConvert('updateTime', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$DictToJson(Dict instance) => <String, dynamic>{
  if (instance.id case final value?) 'id': value,
  if (instance.typeId case final value?) 'typeId': value,
  if (instance.typeCode case final value?) 'typeCode': value,
  if (instance.label case final value?) 'label': value,
  if (instance.value case final value?) 'value': value,
  if (instance.sort case final value?) 'sort': value,
  if (instance.status case final value?) 'status': value,
  if (instance.remark case final value?) 'remark': value,
  if (instance.cssClass case final value?) 'cssClass': value,
  if (instance.listClass case final value?) 'listClass': value,
  if (instance.isDefault case final value?) 'isDefault': value,
  if (instance.createTime case final value?) 'createTime': value,
  if (instance.updateTime case final value?) 'updateTime': value,
};

DictQuery _$DictQueryFromJson(Map<String, dynamic> json) =>
    $checkedCreate('DictQuery', json, ($checkedConvert) {
      final val = DictQuery(
        pageNum: $checkedConvert('pageNum', (v) => (v as num?)?.toInt()),
        pageSize: $checkedConvert('pageSize', (v) => (v as num?)?.toInt()),
        typeCode: $checkedConvert('typeCode', (v) => v as String?),
        keywords: $checkedConvert('keywords', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$DictQueryToJson(DictQuery instance) => <String, dynamic>{
  if (instance.pageNum case final value?) 'pageNum': value,
  if (instance.pageSize case final value?) 'pageSize': value,
  if (instance.typeCode case final value?) 'typeCode': value,
  if (instance.keywords case final value?) 'keywords': value,
};

DictForm _$DictFormFromJson(Map<String, dynamic> json) =>
    $checkedCreate('DictForm', json, ($checkedConvert) {
      final val = DictForm(
        id: $checkedConvert('id', (v) => (v as num?)?.toInt()),
        typeId: $checkedConvert('typeId', (v) => (v as num?)?.toInt()),
        typeCode: $checkedConvert('typeCode', (v) => v as String?),
        label: $checkedConvert('label', (v) => v as String?),
        value: $checkedConvert('value', (v) => v as String?),
        sort: $checkedConvert('sort', (v) => (v as num?)?.toInt()),
        status: $checkedConvert('status', (v) => (v as num?)?.toInt()),
        remark: $checkedConvert('remark', (v) => v as String?),
        cssClass: $checkedConvert('cssClass', (v) => v as String?),
        listClass: $checkedConvert('listClass', (v) => v as String?),
        isDefault: $checkedConvert('isDefault', (v) => (v as num?)?.toInt()),
      );
      return val;
    });

Map<String, dynamic> _$DictFormToJson(DictForm instance) => <String, dynamic>{
  if (instance.id case final value?) 'id': value,
  if (instance.typeId case final value?) 'typeId': value,
  if (instance.typeCode case final value?) 'typeCode': value,
  if (instance.label case final value?) 'label': value,
  if (instance.value case final value?) 'value': value,
  if (instance.sort case final value?) 'sort': value,
  if (instance.status case final value?) 'status': value,
  if (instance.remark case final value?) 'remark': value,
  if (instance.cssClass case final value?) 'cssClass': value,
  if (instance.listClass case final value?) 'listClass': value,
  if (instance.isDefault case final value?) 'isDefault': value,
};

DictOption _$DictOptionFromJson(Map<String, dynamic> json) =>
    $checkedCreate('DictOption', json, ($checkedConvert) {
      final val = DictOption(
        value: $checkedConvert('value', (v) => v as String),
        label: $checkedConvert('label', (v) => v as String),
        children: $checkedConvert(
          'children',
          (v) => (v as List<dynamic>?)
              ?.map((e) => DictOption.fromJson(e as Map<String, dynamic>))
              .toList(),
        ),
      );
      return val;
    });

Map<String, dynamic> _$DictOptionToJson(DictOption instance) =>
    <String, dynamic>{
      'value': instance.value,
      'label': instance.label,
      if (instance.children?.map((e) => e.toJson()).toList() case final value?)
        'children': value,
    };
