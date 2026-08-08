import 'package:dio/dio.dart';

import '../core/constants/api_constants.dart';
import '../core/network/page_result.dart';
import '../models/api_key_model.dart';

/// API 密钥管理服务
///
/// 对齐 JS SDK ApiKeyAPI 全部方法。
class ApiKeyService {
  const ApiKeyService(this._dio);

  final Dio _dio;

  /// 分页查询 API 密钥
  Future<PageResult<ApiKeyVO>> getPage(ApiKeyQuery query) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.authApiKeys}/page',
      queryParameters: query.toQuery(),
    );
    final data = response.data!['data'] as Map<String, dynamic>;
    final list = (data['list'] as List<dynamic>?)
            ?.map(
              (e) => ApiKeyVO.fromJson(e as Map<String, dynamic>),
            )
            .toList() ??
        [];
    return PageResult<ApiKeyVO>(
      list: list,
      total: (data['total'] as int?) ?? 0,
    );
  }

  /// 获取 API 密钥详情
  Future<ApiKeyVO> getById(int id) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.authApiKeys}/$id',
    );
    return ApiKeyVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 创建 API 密钥
  Future<ApiKeyVO> create(ApiKeyCreateForm form) async {
    final response = await _dio.post<Map<String, dynamic>>(
      ApiConstants.authApiKeys,
      data: form.toJson(),
    );
    return ApiKeyVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 更新 API 密钥
  Future<void> update(int id, ApiKeyUpdateForm form) async {
    await _dio.put<Map<String, dynamic>>(
      '${ApiConstants.authApiKeys}/$id',
      data: form.toJson(),
    );
  }

  /// 删除/吊销 API 密钥
  Future<void> delete(int id) async {
    await _dio.delete<Map<String, dynamic>>(
      '${ApiConstants.authApiKeys}/$id',
    );
  }

  /// 吊销 API 密钥（语义化方法，实际调用 delete）
  Future<void> revoke(int id) async {
    await _dio.put<Map<String, dynamic>>(
      '${ApiConstants.authApiKeys}/$id/revoke',
    );
  }
}
