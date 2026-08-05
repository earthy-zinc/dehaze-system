import 'package:dio/dio.dart';

import '../core/constants/api_constants.dart';
import '../models/algorithm_model.dart';

/// 算法服务
///
/// 封装算法管理相关 API：
/// - getAlgorithmList: 获取算法树形列表（移动端取已发布叶子用于选择）
class AlgorithmService {
  const AlgorithmService(this._dio);

  final Dio _dio;

  /// 获取算法树形列表
  ///
  /// GET /algorithms
  Future<List<AlgorithmModel>> getAlgorithmList() async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.algorithm,
    );
    final list = response.data!['data'] as List<dynamic>? ?? [];
    return list
        .map((e) => AlgorithmModel.fromJson(e as Map<String, dynamic>))
        .toList();
  }
}
