import 'package:dio/dio.dart';

import '../core/constants/api_constants.dart';
import '../core/network/page_result.dart';
import '../models/package_model.dart';

/// 套餐 + 优惠券服务
///
/// 封装套餐与优惠券相关 API，包含管理端和用户端。
class PackageService {
  const PackageService(this._dio);

  final Dio _dio;

  // ==================== 套餐（管理端） ====================

  /// 套餐分页列表
  ///
  /// GET /packages/page
  Future<PageResult<PackagePageVO>> getPage(PackageQuery query) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.packagesPage,
      queryParameters: query.toQueryParameters(),
    );
    final data = response.data!['data'] as Map<String, dynamic>;
    final list = (data['list'] as List<dynamic>? ?? [])
        .map((e) => PackagePageVO.fromJson(e as Map<String, dynamic>))
        .toList();
    final total = data['total'] as int? ?? 0;
    return PageResult(list: list, total: total);
  }

  /// 获取套餐详情（管理端/用户端共用）
  ///
  /// GET /packages/{id}
  Future<PackageDetailVO> getById(int id) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.packages}/$id',
    );
    return PackageDetailVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 新增套餐
  ///
  /// POST /packages
  /// 返回新套餐 ID
  Future<int> add(PackageForm form) async {
    final response = await _dio.post<Map<String, dynamic>>(
      ApiConstants.packages,
      data: form.toJson(),
    );
    final data = response.data!['data'];
    if (data is int) return data;
    if (data is Map<String, dynamic>) return data['id'] as int;
    return data as int;
  }

  /// 修改套餐
  ///
  /// PUT /packages/{id}
  Future<void> update(int id, PackageForm form) async {
    await _dio.put<Map<String, dynamic>>(
      '${ApiConstants.packages}/$id',
      data: form.toJson(),
    );
  }

  /// 删除套餐
  ///
  /// DELETE /packages/{id}
  Future<void> delete(int id) async {
    await _dio.delete<Map<String, dynamic>>(
      '${ApiConstants.packages}/$id',
    );
  }

  /// 上下架套餐
  ///
  /// PUT /packages/{id}/status
  Future<void> updateStatus(int id, int status) async {
    await _dio.put<Map<String, dynamic>>(
      '${ApiConstants.packages}/$id/status',
      queryParameters: {'status': status},
    );
  }

  /// 销售统计
  ///
  /// GET /packages/sales/stats
  Future<SalesStatsVO> getSalesStats() async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.packagesSalesStats,
    );
    return SalesStatsVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  // ==================== 套餐（用户端） ====================

  /// 在售套餐列表
  ///
  /// GET /packages
  Future<List<PackageDetailVO>> getList() async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.packages,
    );
    final list = response.data!['data'] as List<dynamic>? ?? [];
    return list
        .map((e) => PackageDetailVO.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  /// 套餐详情（用户端，与 getById 同接口）
  ///
  /// GET /packages/{id}
  Future<PackageDetailVO> getDetail(int id) async {
    return getById(id);
  }

  /// 价格计算（下单前预览）
  ///
  /// GET /packages/calculate-price?packageId=...&couponId=...
  Future<PriceResult> calculatePrice(int packageId, {int? couponId}) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.packagesCalculatePrice,
      queryParameters: {
        'packageId': packageId,
        if (couponId != null) 'couponId': couponId,
      },
    );
    return PriceResult.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  // ==================== 优惠券（管理端） ====================

  /// 优惠券分页列表
  ///
  /// GET /packages/coupons/page
  Future<PageResult<CouponVO>> couponGetPage(CouponQuery query) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.packagesCouponsPage,
      queryParameters: query.toQueryParameters(),
    );
    final data = response.data!['data'] as Map<String, dynamic>;
    final list = (data['list'] as List<dynamic>? ?? [])
        .map((e) => CouponVO.fromJson(e as Map<String, dynamic>))
        .toList();
    final total = data['total'] as int? ?? 0;
    return PageResult(list: list, total: total);
  }

  /// 获取优惠券详情
  ///
  /// GET /packages/coupons/{id}
  Future<CouponVO> couponGetById(int id) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.packagesCoupons}/$id',
    );
    return CouponVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 创建优惠券
  ///
  /// POST /packages/coupons
  /// 返回优惠券 ID
  Future<int> couponAdd(CouponForm form) async {
    final response = await _dio.post<Map<String, dynamic>>(
      ApiConstants.packagesCoupons,
      data: form.toJson(),
    );
    final data = response.data!['data'];
    if (data is int) return data;
    if (data is Map<String, dynamic>) return data['id'] as int;
    return data as int;
  }

  /// 修改优惠券
  ///
  /// PUT /packages/coupons/{id}
  Future<void> couponUpdate(int id, CouponForm form) async {
    await _dio.put<Map<String, dynamic>>(
      '${ApiConstants.packagesCoupons}/$id',
      data: form.toJson(),
    );
  }

  /// 删除优惠券
  ///
  /// DELETE /packages/coupons/{id}
  Future<void> couponDelete(int id) async {
    await _dio.delete<Map<String, dynamic>>(
      '${ApiConstants.packagesCoupons}/$id',
    );
  }

  /// 批量发放优惠券
  ///
  /// POST /packages/coupons/batch
  /// 返回发放成功数量
  Future<int> couponBatchDistribute(CouponBatchDistributeForm form) async {
    final response = await _dio.post<Map<String, dynamic>>(
      ApiConstants.packagesCouponsBatch,
      data: form.toJson(),
    );
    final data = response.data!['data'];
    if (data is int) return data;
    if (data is Map<String, dynamic>) return data['successCount'] as int? ?? 0;
    return 0;
  }

  // ==================== 优惠券（用户端） ====================

  /// 我的优惠券列表
  ///
  /// GET /packages/coupons/my
  Future<List<UserCouponVO>> getMyCoupons({int? status}) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.packagesCouponsMy,
      queryParameters: {
        if (status != null) 'status': status,
      },
    );
    final list = response.data!['data'] as List<dynamic>? ?? [];
    return list
        .map((e) => UserCouponVO.fromJson(e as Map<String, dynamic>))
        .toList();
  }
}
