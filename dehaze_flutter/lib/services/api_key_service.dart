import 'package:dio/dio.dart';

import '../core/constants/api_constants.dart';
import '../models/api_key_model.dart';

/// API 密钥管理服务
///
/// 对齐 JS SDK ApiKeyAPI：创建 / 列表 / 删除。
class ApiKeyService {
  const ApiKeyService(this._dio);

  final Dio _dio;

  /// 创建 API 密钥
  ///
  /// 返回体中的 `apiKey` 为完整密钥，仅此一次返回。
  Future<ApiKeyVO> create(ApiKeyCreateForm form) async {
    final response = await _dio.post<Map<String, dynamic>>(
      ApiConstants.authApiKeys,
      data: form.toJson(),
    );
    return ApiKeyVO.fromJson(response.data!['data'] as Map<String, dynamic>);
  }

  /// 查询当前用户的 API 密钥列表（后端返回数组，非分页）
  Future<List<ApiKeyVO>> list() async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.authApiKeys,
    );
    final list = response.data!['data'] as List<dynamic>? ?? [];
    return list
        .map((e) => ApiKeyVO.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  /// 删除 API 密钥
  Future<void> delete(int id) async {
    await _dio.delete<Map<String, dynamic>>('${ApiConstants.authApiKeys}/$id');
  }
}
