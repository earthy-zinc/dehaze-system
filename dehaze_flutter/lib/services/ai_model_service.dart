import 'package:dio/dio.dart';

import '../core/constants/api_constants.dart';
import '../models/ai_model_model.dart';

/// AI 模型目录服务
///
/// 消费后端启用模型列表（按登录用户 VIP 等级过滤），供图像模型选择器
/// 与展示名解析使用。
class AiModelService {
  const AiModelService(this._dio);

  final Dio _dio;

  /// 启用模型列表（无特殊权限）
  ///
  /// GET /api/v1/ai/models/enabled?modelType=image_edit
  Future<List<AiModelVO>> listEnabledModels({
    String? modelType,
    String? taskTags,
  }) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.aiModelsEnabled,
      queryParameters: {'modelType': ?modelType, 'taskTags': ?taskTags},
    );
    final list = response.data!['data'] as List<dynamic>? ?? [];
    return list
        .map((e) => AiModelVO.fromJson(e as Map<String, dynamic>))
        .toList();
  }
}
