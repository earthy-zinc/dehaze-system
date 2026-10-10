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
  'status': ?instance.status,
  'remark': ?instance.remark,
  'createTime': ?instance.createTime,
  'updateTime': ?instance.updateTime,
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
      'pageNum': ?instance.pageNum,
      'pageSize': ?instance.pageSize,
      'keywords': ?instance.keywords,
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
      'id': ?instance.id,
      'name': ?instance.name,
      'code': ?instance.code,
      'status': instance.status,
      'remark': ?instance.remark,
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
  'id': ?instance.id,
  'typeId': ?instance.typeId,
  'typeCode': ?instance.typeCode,
  'label': ?instance.label,
  'value': ?instance.value,
  'sort': ?instance.sort,
  'status': ?instance.status,
  'remark': ?instance.remark,
  'cssClass': ?instance.cssClass,
  'listClass': ?instance.listClass,
  'isDefault': ?instance.isDefault,
  'createTime': ?instance.createTime,
  'updateTime': ?instance.updateTime,
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
  'pageNum': ?instance.pageNum,
  'pageSize': ?instance.pageSize,
  'typeCode': ?instance.typeCode,
  'keywords': ?instance.keywords,
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
  'id': ?instance.id,
  'typeId': ?instance.typeId,
  'typeCode': ?instance.typeCode,
  'label': ?instance.label,
  'value': ?instance.value,
  'sort': ?instance.sort,
  'status': ?instance.status,
  'remark': ?instance.remark,
  'cssClass': ?instance.cssClass,
  'listClass': ?instance.listClass,
  'isDefault': ?instance.isDefault,
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
      'children': ?instance.children?.map((e) => e.toJson()).toList(),
    };
