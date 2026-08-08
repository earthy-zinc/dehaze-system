import 'dart:typed_data';

import 'package:dio/dio.dart';

import '../core/constants/api_constants.dart';
import '../core/network/page_result.dart';
import '../models/file_model.dart';

/// 文件服务
///
/// 对齐 JS SDK FileAPI 的全部方法。
/// 封装文件上传、秒传检查、分页查询、详情获取、下载等 API。
class FileService {
  const FileService(this._dio);

  final Dio _dio;

  /// 从字节流上传文件
  Future<FileInfo> uploadBytes(
    Uint8List bytes,
    String fileName, {
    void Function(int sent, int total)? onProgress,
  }) async {
    final formData = FormData.fromMap({
      'file': MultipartFile.fromBytes(
        bytes,
        filename: fileName,
      ),
    });

    final response = await _dio.post<Map<String, dynamic>>(
      ApiConstants.files,
      data: formData,
      onSendProgress: onProgress,
    );
    return FileInfo.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// MD5 秒传检查 — 检查文件是否已存在，避免重复上传
  ///
  /// GET ApiConstants.filesCheck, query: { md5 }
  /// 返回 { exists: bool, fileId?: int, url?: string }
  Future<Map<String, dynamic>> check(String md5) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.filesCheck,
      queryParameters: {'md5': md5},
    );
    return response.data!['data'] as Map<String, dynamic>;
  }

  /// 分页查询文件列表（管理端）
  ///
  /// GET ApiConstants.filesPage, query: FileQuery
  Future<PageResult<FileInfo>> getPage([FileQuery? query]) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.filesPage,
      queryParameters: query?.toJson(),
    );
    final data = response.data!['data'] as Map<String, dynamic>;
    final list = (data['list'] as List<dynamic>)
        .map((e) => FileInfo.fromJson(e as Map<String, dynamic>))
        .toList();
    return PageResult(list: list, total: data['total'] as int);
  }

  /// 根据 ID 获取文件信息
  ///
  /// GET ${ApiConstants.files}/$id
  Future<FileInfo> getById(int id) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.files}/$id',
    );
    return FileInfo.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 下载文件（返回流式响应，用于保存到本地或分享）
  ///
  /// GET ApiConstants.filesDownload/$id
  /// 使用 ResponseType.stream 获取文件流
  Future<Response<List<int>>> download(int id) async {
    return _dio.get(
      '${ApiConstants.filesDownload}/$id',
      options: Options(responseType: ResponseType.stream),
    );
  }
}
