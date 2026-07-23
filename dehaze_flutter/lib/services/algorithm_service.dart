import 'package:dio/dio.dart';

import '../core/constants/api_constants.dart';
import '../models/algorithm_model.dart';

/// 算法服务
///
/// 封装算法管理相关 API：
/// - getAlgorithmOptions: 获取算法下拉选项
/// - getAlgorithmList: 获取算法树形列表
/// - getAlgorithmDetail: 获取算法详情
class AlgorithmService {
  const AlgorithmService(this._dio);

  final Dio _dio;

  /// 获取算法下拉选项
  ///
  /// GET /algorithm/options
  Future<List<AlgorithmOption>> getAlgorithmOptions() async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.algorithmOptions,
    );
    // ResponseInterceptor 已保证 code=='00000'，失败已 reject 为 ApiException
    final list = response.data!['data'] as List<dynamic>? ?? [];
    return list
        .map((e) => AlgorithmOption.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  /// 获取算法树形列表
  ///
  /// GET /algorithm
  Future<List<AlgorithmModel>> getAlgorithmList() async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.algorithm,
    );
    final list = response.data!['data'] as List<dynamic>? ?? [];
    return list
        .map((e) => AlgorithmModel.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  /// 获取算法详情
  ///
  /// GET /algorithm/{id}
  Future<AlgorithmModel> getAlgorithmDetail(int id) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.algorithm}/$id',
    );
    return AlgorithmModel.fromJson(response.data!['data'] as Map<String, dynamic>);
  }
}
