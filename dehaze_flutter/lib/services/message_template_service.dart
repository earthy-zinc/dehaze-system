import 'package:dio/dio.dart';

import '../core/constants/api_constants.dart';
import '../core/network/page_result.dart';
import '../models/message_template_model.dart';

class MessageTemplateService {
  const MessageTemplateService(this._dio);

  final Dio _dio;

  /// 分页查询消息模板
  Future<PageResult<MessageTemplateVO>> getPage(
    MessageTemplateQuery query,
  ) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.messageTemplatesPage,
      queryParameters: query.toQuery(),
    );
    return PageResult.fromResponse(
      response.data!['data'] as Map<String, dynamic>,
      MessageTemplateVO.fromJson,
    );
  }

  /// 获取模板详情
  Future<MessageTemplateVO> getById(int id) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.messageTemplates}/$id',
    );
    return MessageTemplateVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 根据编码获取模板
  Future<MessageTemplateVO> getByCode(String code) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.messageTemplates}/code/$code',
    );
    return MessageTemplateVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 新增模板，返回模板 ID
  Future<int> add(MessageTemplateForm form) async {
    final response = await _dio.post<Map<String, dynamic>>(
      ApiConstants.messageTemplates,
      data: form.toJson(),
    );
    return (response.data!['data'] as Map<String, dynamic>)['id'] as int;
  }

  /// 更新模板
  Future<void> update(int id, MessageTemplateForm form) async {
    await _dio.put<Map<String, dynamic>>(
      '${ApiConstants.messageTemplates}/$id',
      data: form.toJson(),
    );
  }

  /// 删除模板
  Future<void> delete(int id) async {
    await _dio.delete<Map<String, dynamic>>(
      '${ApiConstants.messageTemplates}/$id',
    );
  }
}
