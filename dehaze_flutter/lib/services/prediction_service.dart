import 'package:dio/dio.dart';

import '../core/constants/api_constants.dart';
import '../core/network/page_result.dart';
import '../models/prediction_model.dart';

/// 预测服务
///
/// 封装模型预测相关 API：
/// - predict: 执行模型预测（同步返回结果）
/// - getPredictionLogs: 获取预测日志列表
class PredictionService {
  const PredictionService(this._dio);

  final Dio _dio;

  /// 执行模型预测
  ///
  /// POST /prediction
  Future<PredictionResponse> predict(PredictionRequest request) async {
    final response = await _dio.post<Map<String, dynamic>>(
      ApiConstants.prediction,
      data: request.toJson(),
    );

    final result = response.data!;
    if (result['code']?.toString() == ApiConstants.successCode) {
      return PredictionResponse.fromJson(
        result['data'] as Map<String, dynamic>,
      );
    }
    throw Exception(result['msg'] ?? '预测请求失败');
  }

  /// 获取预测日志列表
  ///
  /// GET /prediction/logs
  Future<PageResult<PredictionLog>> getPredictionLogs({
    int pageNum = 1,
    int pageSize = 10,
  }) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.predictionLogs,
      queryParameters: {
        'pageNum': pageNum,
        'pageSize': pageSize,
      },
    );

    final result = response.data!;
    if (result['code']?.toString() == ApiConstants.successCode) {
      final data = result['data'] as Map<String, dynamic>;
      final list = (data['list'] as List<dynamic>? ?? [])
          .map((e) => PredictionLog.fromJson(e as Map<String, dynamic>))
          .toList();
      final total = data['total'] as int? ?? 0;
      return PageResult(list: list, total: total);
    }
    throw Exception(result['msg'] ?? '获取预测日志失败');
  }
}
