import 'package:dio/dio.dart';

import '../core/constants/api_constants.dart';
import '../models/billing_model.dart';

/// AI 计费服务
///
/// 用户端积分余额查询（展示口径，不参与提交前校验）。
class BillingService {
  const BillingService(this._dio);

  final Dio _dio;

  /// 用户余额查询
  ///
  /// GET /api/v1/ai-billing/balance
  Future<BalanceVO> getBalance() async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.aiBillingBalance,
    );
    return BalanceVO.fromJson(response.data!['data'] as Map<String, dynamic>);
  }
}
