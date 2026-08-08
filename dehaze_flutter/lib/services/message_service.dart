import 'package:dio/dio.dart';

import '../core/constants/api_constants.dart';
import '../models/message_model.dart';

class MessageService {
  const MessageService(this._dio);

  final Dio _dio;

  Future<Map<String, dynamic>> getPage({
    int pageNum = 1,
    int pageSize = 20,
    String? type,
    int? readStatus,
  }) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.messages,
      queryParameters: {
        'pageNum': pageNum,
        'pageSize': pageSize,
        if (type != null) 'type': type,
        if (readStatus != null) 'readStatus': readStatus,
      },
    );
    return response.data!;
  }

  Future<UnreadCountVO> getUnreadCount() async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.messagesUnreadCount,
    );
    return UnreadCountVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  Future<MessageVO> getDetail(int id) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.messages}/$id',
    );
    return MessageVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  Future<void> markRead(int id) async {
    await _dio.patch<Map<String, dynamic>>('${ApiConstants.messages}/$id/_read');
  }

  Future<ReadAllResult> markAllRead({String? type}) async {
    final response = await _dio.patch<Map<String, dynamic>>(
      ApiConstants.messagesReadAll,
      queryParameters: type != null ? {'type': type} : null,
    );
    return ReadAllResult.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  Future<void> deleteByIds(String ids) async {
    await _dio.delete<Map<String, dynamic>>('${ApiConstants.messages}/$ids');
  }

  Future<Map<String, dynamic>> search({
    required String keyword,
    int pageNum = 1,
    int pageSize = 20,
  }) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.messagesSearch,
      queryParameters: {
        'keyword': keyword,
        'pageNum': pageNum,
        'pageSize': pageSize,
      },
    );
    return response.data!;
  }

  Future<void> send(Map<String, dynamic> data) async {
    await _dio.post<Map<String, dynamic>>(ApiConstants.messagesSend, data: data);
  }
}
