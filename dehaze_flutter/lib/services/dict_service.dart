import 'package:dio/dio.dart';

import '../core/constants/api_constants.dart';
import '../core/network/page_result.dart';
import '../models/dict_model.dart';

/// 字典服务
///
/// 封装字典类型和字典项的 CRUD 操作。
/// 对应后端接口：
/// - 字典类型：/api/v1/dict/types/**
/// - 字典项：/api/v1/dict/**
class DictService {
  const DictService(this._dio);

  final Dio _dio;

  // ==================== 字典类型 ====================

  /// 字典类型分页列表
  Future<PageResult<DictType>> getTypePage(DictTypeQuery query) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.dictTypesPage,
      queryParameters: query.toQuery(),
    );
    final data = response.data!['data'] as Map<String, dynamic>;
    return PageResult<DictType>(
      list: (data['list'] as List<dynamic>)
          .map((e) => DictType.fromJson(e as Map<String, dynamic>))
          .toList(),
      total: (data['total'] as num).toInt(),
    );
  }

  /// 字典类型列表（不分页）
  Future<List<DictType>> getTypeList() async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.dictTypes,
    );
    final list = response.data!['data'] as List<dynamic>? ?? [];
    return list
        .map((e) => DictType.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  /// 字典类型表单数据（编辑回显用）
  Future<DictTypeForm> getTypeForm(int id) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.dictTypes}/$id/form',
    );
    return DictTypeForm.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 新增字典类型
  Future<int> addType(DictTypeForm form) async {
    final response = await _dio.post<Map<String, dynamic>>(
      ApiConstants.dictTypes,
      data: form.toJson(),
    );
    return (response.data!['data'] as num).toInt();
  }

  /// 修改字典类型
  Future<void> updateType(int id, DictTypeForm form) async {
    await _dio.put<Map<String, dynamic>>(
      '${ApiConstants.dictTypes}/$id',
      data: form.toJson(),
    );
  }

  /// 删除字典类型
  Future<void> deleteType(int id) async {
    await _dio.delete<Map<String, dynamic>>(
      '${ApiConstants.dictTypes}/$id',
    );
  }

  // ==================== 字典项 ====================

  /// 字典项分页列表
  Future<PageResult<Dict>> getDictPage(DictQuery query) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.dictPage,
      queryParameters: query.toQuery(),
    );
    final data = response.data!['data'] as Map<String, dynamic>;
    return PageResult<Dict>(
      list: (data['list'] as List<dynamic>)
          .map((e) => Dict.fromJson(e as Map<String, dynamic>))
          .toList(),
      total: (data['total'] as num).toInt(),
    );
  }

  /// 字典项表单数据（编辑回显用）
  Future<DictForm> getDictForm(int id) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.dict}/$id/form',
    );
    return DictForm.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 获取字典类型的选项（下拉框数据）
  Future<List<DictOption>> getDictOptions(String typeCode) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.dict}/$typeCode/options',
    );
    final list = response.data!['data'] as List<dynamic>? ?? [];
    return list
        .map((e) => DictOption.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  /// 新增字典项
  Future<int> addDict(DictForm form) async {
    final response = await _dio.post<Map<String, dynamic>>(
      ApiConstants.dict,
      data: form.toJson(),
    );
    return (response.data!['data'] as num).toInt();
  }

  /// 修改字典项
  Future<void> updateDict(int id, DictForm form) async {
    await _dio.put<Map<String, dynamic>>(
      '${ApiConstants.dict}/$id',
      data: form.toJson(),
    );
  }

  /// 删除字典项
  Future<void> deleteDict(int id) async {
    await _dio.delete<Map<String, dynamic>>(
      '${ApiConstants.dict}/$id',
    );
  }
}
