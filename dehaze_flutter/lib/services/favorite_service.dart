import 'package:dio/dio.dart';

import '../core/constants/api_constants.dart';
import '../core/network/page_result.dart';
import '../models/favorite_model.dart';

/// 收藏管理服务
///
/// 为算法、数据集等业务实体提供统一的收藏能力。
/// 对齐 JS SDK FavoriteAPI 全部方法。
class FavoriteService {
  const FavoriteService(this._dio);

  final Dio _dio;

  /// 收藏列表分页查询（支持类型筛选、排序）
  Future<PageResult<FavoriteVO>> getPage(FavoriteQuery query) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.favoritesPage,
      queryParameters: query.toQuery(),
    );
    final data = response.data!['data'] as Map<String, dynamic>;
    final list = (data['list'] as List<dynamic>?)
            ?.map((e) => FavoriteVO.fromJson(e as Map<String, dynamic>))
            .toList() ??
        [];
    return PageResult<FavoriteVO>(
      list: list,
      total: (data['total'] as int?) ?? 0,
    );
  }

  /// 添加收藏（同一用户对同一对象只能收藏一次）
  Future<void> add(FavoriteForm form) async {
    await _dio.post<Map<String, dynamic>>(
      ApiConstants.favorites,
      data: form.toJson(),
    );
  }

  /// 取消收藏
  Future<void> remove(int targetId, String targetType) async {
    await _dio.delete<Map<String, dynamic>>(
      ApiConstants.favorites,
      queryParameters: {
        'targetId': targetId,
        'targetType': targetType,
      },
    );
  }

  /// 检查指定对象是否已收藏（用于前端图标状态判断）
  Future<FavoriteStatus> check(int targetId, String targetType) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.favorites}/check',
      queryParameters: {
        'targetId': targetId,
        'targetType': targetType,
      },
    );
    return FavoriteStatus.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 收藏数量统计
  Future<FavoriteCount> count(int targetId, String targetType) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.favoritesCount,
      queryParameters: {
        'targetId': targetId,
        'targetType': targetType,
      },
    );
    return FavoriteCount.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 切换收藏状态（已收藏则取消，未收藏则添加）
  Future<void> toggle(int targetId, String targetType) async {
    await _dio.post<Map<String, dynamic>>(
      '${ApiConstants.favorites}/toggle',
      data: {
        'targetId': targetId,
        'targetType': targetType,
      },
    );
  }
}
