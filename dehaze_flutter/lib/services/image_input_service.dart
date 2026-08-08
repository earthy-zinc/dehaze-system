import 'package:dio/dio.dart';

import '../core/constants/api_constants.dart';
import '../core/network/page_result.dart';
import '../models/image_input_model.dart';

/// 图片输入历史记录服务
///
/// 对齐 JS SDK ImageInputHistoryAPI 全部方法。
class ImageInputService {
  const ImageInputService(this._dio);

  final Dio _dio;

  /// 分页查询历史记录
  Future<PageResult<ImageInputHistoryVO>> getPage(
    ImageInputHistoryQuery query,
  ) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.imageInputHistory,
      queryParameters: query.toQuery(),
    );
    final data = response.data!['data'] as Map<String, dynamic>;
    final list = (data['list'] as List<dynamic>?)
            ?.map(
              (e) =>
                  ImageInputHistoryVO.fromJson(e as Map<String, dynamic>),
            )
            .toList() ??
        [];
    return PageResult<ImageInputHistoryVO>(
      list: list,
      total: (data['total'] as int?) ?? 0,
    );
  }

  /// 获取历史记录详情
  Future<ImageInputHistoryVO> getById(int id) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.imageInputHistory}/$id',
    );
    return ImageInputHistoryVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 创建历史记录
  Future<int> create(Map<String, dynamic> form) async {
    final response = await _dio.post<Map<String, dynamic>>(
      ApiConstants.imageInputHistory,
      data: form,
    );
    return response.data!['data'] as int;
  }

  /// 更新历史记录
  Future<void> update(int id, Map<String, dynamic> form) async {
    await _dio.put<Map<String, dynamic>>(
      '${ApiConstants.imageInputHistory}/$id',
      data: form,
    );
  }

  /// 删除单条历史记录
  Future<void> delete(int id) async {
    await _dio.delete<Map<String, dynamic>>(
      '${ApiConstants.imageInputHistory}/$id',
    );
  }

  /// 批量删除历史记录
  Future<void> batchDelete(ImageInputHistoryBatchDeleteForm form) async {
    await _dio.delete<Map<String, dynamic>>(
      ApiConstants.imageInputHistoryBatch,
      data: form.toJson(),
    );
  }

  /// 清空所有历史记录
  Future<void> clear() async {
    await _dio.delete<Map<String, dynamic>>(
      ApiConstants.imageInputHistoryClear,
    );
  }

  /// 同步本地与云端历史记录
  Future<void> sync(ImageInputHistorySyncForm form) async {
    await _dio.post<Map<String, dynamic>>(
      ApiConstants.imageInputHistorySync,
      data: form.toJson(),
    );
  }
}
