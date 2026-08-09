import 'package:dio/dio.dart';

import '../core/constants/api_constants.dart';
import '../core/network/page_result.dart';
import '../models/announcement_model.dart';

class AnnouncementService {
  const AnnouncementService(this._dio);

  final Dio _dio;

  // ==================== 管理端 ====================

  /// 分页查询公告
  Future<PageResult<AnnouncementVO>> getPage(AnnouncementQuery query) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.announcementsPage,
      queryParameters: query.toQuery(),
    );
    return PageResult.fromResponse(
      response.data!['data'] as Map<String, dynamic>,
      AnnouncementVO.fromJson,
    );
  }

  /// 获取公告详情
  Future<AnnouncementVO> getById(int id) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.announcements}/$id',
    );
    return AnnouncementVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 新增公告，返回公告 ID
  Future<int> add(AnnouncementForm form) async {
    final response = await _dio.post<Map<String, dynamic>>(
      ApiConstants.announcements,
      data: form.toJson(),
    );
    return (response.data!['data'] as Map<String, dynamic>)['id'] as int;
  }

  /// 更新公告
  Future<void> update(int id, AnnouncementForm form) async {
    await _dio.put<Map<String, dynamic>>(
      '${ApiConstants.announcements}/$id',
      data: form.toJson(),
    );
  }

  /// 删除公告
  Future<void> delete(int id) async {
    await _dio.delete<Map<String, dynamic>>(
      '${ApiConstants.announcements}/$id',
    );
  }

  /// 发送公告
  Future<AnnouncementSendResult> send(int id) async {
    final response = await _dio.post<Map<String, dynamic>>(
      '${ApiConstants.announcements}/$id/send',
    );
    return AnnouncementSendResult.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  // ==================== 用户端 ====================

  /// 获取当前生效的公告列表
  Future<List<AnnouncementVO>> getActiveAnnouncements() async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.announcements,
    );
    final data = response.data!['data'] as List<dynamic>;
    return data
        .map((e) => AnnouncementVO.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  /// 标记公告已读
  Future<void> markAsRead(int id) async {
    await _dio.post<Map<String, dynamic>>(
      '${ApiConstants.announcements}/$id/read',
    );
  }
}
