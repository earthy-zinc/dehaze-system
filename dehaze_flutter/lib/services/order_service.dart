import 'package:dio/dio.dart';

import '../core/constants/api_constants.dart';
import '../core/network/page_result.dart';
import '../models/order_model.dart';

/// 订单服务（用户端 + 管理端）
///
/// 对齐 JS SDK OrderAPI 的全部方法。
class OrderService {
  const OrderService(this._dio);

  final Dio _dio;

  // ==================== 用户端 ====================

  /// 创建订单
  Future<MyOrderVO> create(OrderCreateForm form) async {
    final response = await _dio.post<Map<String, dynamic>>(
      ApiConstants.orders,
      data: form.toJson(),
    );
    return MyOrderVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 我的订单列表
  Future<PageResult<MyOrderVO>> getMyOrders(MyOrderQuery query) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.ordersMy,
      queryParameters: query.toJson(),
    );
    final data = response.data!['data'] as Map<String, dynamic>;
    return PageResult<MyOrderVO>(
      list: (data['list'] as List<dynamic>? ?? [])
          .map((e) => MyOrderVO.fromJson(e as Map<String, dynamic>))
          .toList(),
      total: data['total'] as int? ?? 0,
    );
  }

  /// 我的订单详情
  Future<OrderDetailVO> getMyOrderDetail(int id) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.ordersMy}/$id',
    );
    return OrderDetailVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 取消订单
  Future<void> cancelOrder(int id) async {
    await _dio.put<Map<String, dynamic>>(
      '${ApiConstants.orders}/$id/cancel',
    );
  }

  /// 发起支付
  Future<PayResult> pay(PayRequest request) async {
    final response = await _dio.post<Map<String, dynamic>>(
      '${ApiConstants.orders}/pay',
      data: request.toJson(),
    );
    return PayResult.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 查询自动续费配置
  Future<AutoRenewConfigVO> getAutoRenewConfig(int orderId) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.ordersAutoRenewConfig}/$orderId',
    );
    return AutoRenewConfigVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 修改自动续费配置
  Future<void> updateAutoRenewConfig(AutoRenewConfigForm form) async {
    await _dio.put<Map<String, dynamic>>(
      ApiConstants.ordersAutoRenewConfig,
      data: form.toJson(),
    );
  }

  // ==================== 管理端 ====================

  /// 订单分页列表
  Future<PageResult<OrderPageVO>> getPage(OrderQuery query) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.ordersPage,
      queryParameters: query.toJson(),
    );
    final data = response.data!['data'] as Map<String, dynamic>;
    return PageResult<OrderPageVO>(
      list: (data['list'] as List<dynamic>? ?? [])
          .map((e) => OrderPageVO.fromJson(e as Map<String, dynamic>))
          .toList(),
      total: data['total'] as int? ?? 0,
    );
  }

  /// 订单详情
  Future<OrderDetailVO> getDetail(int id) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.orders}/$id',
    );
    return OrderDetailVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 订单统计
  Future<OrderStatsVO> getStats() async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.ordersStats,
    );
    return OrderStatsVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 退款分页列表
  Future<PageResult<RefundRecordVO>> getRefundPage(RefundQuery query) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.ordersRefundsPage,
      queryParameters: query.toJson(),
    );
    final data = response.data!['data'] as Map<String, dynamic>;
    return PageResult<RefundRecordVO>(
      list: (data['list'] as List<dynamic>? ?? [])
          .map((e) => RefundRecordVO.fromJson(e as Map<String, dynamic>))
          .toList(),
      total: data['total'] as int? ?? 0,
    );
  }

  /// 退款审核
  Future<void> auditRefund(RefundAuditForm form) async {
    await _dio.put<Map<String, dynamic>>(
      ApiConstants.ordersRefunds,
      data: form.toJson(),
    );
  }
}
