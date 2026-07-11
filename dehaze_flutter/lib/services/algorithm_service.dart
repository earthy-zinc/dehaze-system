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

    final result = response.data!;
    if (result['code']?.toString() == ApiConstants.successCode) {
      final list = result['data'] as List<dynamic>? ?? [];
      return list
          .map((e) => AlgorithmOption.fromJson(e as Map<String, dynamic>))
          .toList();
    }
    throw Exception(result['msg'] ?? '获取算法选项失败');
  }

  /// 获取算法树形列表
  ///
  /// GET /algorithm
  Future<List<AlgorithmModel>> getAlgorithmList() async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.algorithm,
    );

    final result = response.data!;
    if (result['code']?.toString() == ApiConstants.successCode) {
      final list = result['data'] as List<dynamic>? ?? [];
      return list
          .map((e) => AlgorithmModel.fromJson(e as Map<String, dynamic>))
          .toList();
    }
    throw Exception(result['msg'] ?? '获取算法列表失败');
  }

  /// 获取算法详情
  ///
  /// GET /algorithm/{id}
  Future<AlgorithmModel> getAlgorithmDetail(int id) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.algorithm}/$id',
    );

    final result = response.data!;
    if (result['code']?.toString() == ApiConstants.successCode) {
      return AlgorithmModel.fromJson(result['data'] as Map<String, dynamic>);
    }
    throw Exception(result['msg'] ?? '获取算法详情失败');
  }
}
