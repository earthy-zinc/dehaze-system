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
  ///
  /// 创建订单不发起渠道下单，第三方支付参数由 [pay] 获取。
  Future<PayResult> create(OrderCreateForm form) async {
    final response = await _dio.post<Map<String, dynamic>>(
      ApiConstants.orders,
      data: form.toJson(),
    );
    return PayResult.fromJson(response.data!['data'] as Map<String, dynamic>);
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

  /// 订单详情（用户端 / 后台共用，按订单号查询）
  Future<OrderDetailVO> getDetail(String orderNo) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.orders}/$orderNo',
    );
    return OrderDetailVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 取消订单（取消原因必填）
  Future<void> cancelOrder(String orderNo, String reason) async {
    await _dio.put<Map<String, dynamic>>(
      '${ApiConstants.orders}/$orderNo/cancel',
      queryParameters: {'reason': reason},
    );
  }

  /// 申请退款
  Future<void> applyRefund(String orderNo, RefundApplyForm form) async {
    await _dio.post<Map<String, dynamic>>(
      '${ApiConstants.orders}/$orderNo/refund',
      data: form.toJson(),
    );
  }

  /// 发起支付
  ///
  /// 订单号在路径中传递，请求体仅描述支付方式。
  Future<PayResult> pay(String orderNo, PayRequest request) async {
    final response = await _dio.post<Map<String, dynamic>>(
      '${ApiConstants.orders}/$orderNo/pay',
      data: request.toJson(),
    );
    return PayResult.fromJson(response.data!['data'] as Map<String, dynamic>);
  }

  /// 查询自动续费配置（按套餐查询）
  Future<AutoRenewConfigVO> getAutoRenewConfig(int packageId) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.ordersAutoRenewConfig,
      queryParameters: {'packageId': packageId},
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

  /// 退款审核（通过 / 驳回由 [RefundAuditForm.approved] 决定，分别对应两个端点）
  Future<void> auditRefund(int refundId, RefundAuditForm form) async {
    final action = form.approved ? 'approve' : 'reject';
    await _dio.put<Map<String, dynamic>>(
      '${ApiConstants.ordersRefunds}/$refundId/$action',
      data: form.toJson(),
    );
  }
}
