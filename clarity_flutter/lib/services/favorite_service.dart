import 'package:dio/dio.dart';

import '../core/constants/api_constants.dart';
import '../core/network/page_result.dart';
import '../models/favorite_model.dart';

/// 收藏管理服务
///
/// 为模型、数据集等业务实体提供统一的收藏能力。
/// 对齐 JS SDK FavoriteAPI 全部方法。
class FavoriteService {
  const FavoriteService(this._dio);

  final Dio _dio;

  /// 收藏列表分页查询（支持类型筛选、关键词搜索、排序）
  Future<PageResult<FavoriteVO>> getPage([FavoriteQuery? query]) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.favoritesPage,
      queryParameters: query?.toQuery(),
    );
    final data = response.data!['data'] as Map<String, dynamic>;
    final list =
        (data['list'] as List<dynamic>?)
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

  /// 批量取消收藏（入参为收藏记录 ID，非业务对象 ID）
  Future<void> deleteByIds(List<int> ids) async {
    await _dio.delete<Map<String, dynamic>>(
      '${ApiConstants.favorites}/${ids.join(',')}',
    );
  }

  /// 检查指定对象是否已收藏（用于前端图标状态判断）
  Future<FavoriteStatus> getStatus(
    int targetId,
    FavoriteTargetType targetType,
  ) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.favorites}/$targetId/status',
      queryParameters: {'targetType': targetType.wireValue},
    );
    return FavoriteStatus.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 收藏数量统计（按类型分组）
  ///
  /// 不传 [targetType] 时返回全部类型的计数，求和即为当前用户收藏总数。
  Future<List<FavoriteCount>> count([FavoriteTargetType? targetType]) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.favoritesCount,
      queryParameters: targetType == null
          ? null
          : {'targetType': targetType.wireValue},
    );
    final list = response.data!['data'] as List<dynamic>? ?? [];
    return list
        .map((e) => FavoriteCount.fromJson(e as Map<String, dynamic>))
        .toList();
  }
}
