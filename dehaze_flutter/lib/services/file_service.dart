import 'package:dio/dio.dart';

import '../core/constants/api_constants.dart';
import '../models/file_model.dart';

/// 文件服务
///
/// 封装文件管理相关 API：
/// - uploadFile: 上传文件
/// - checkMd5: MD5 秒传检查
/// - getFileDetail: 获取文件详情
/// - downloadFile: 下载文件
class FileService {
  const FileService(this._dio);

  final Dio _dio;

  /// 上传文件
  ///
  /// POST /files
  /// Content-Type: multipart/form-data
  Future<FileUploadResponse> uploadFile(
    String filePath, {
    String? fileName,
    void Function(int sent, int total)? onProgress,
  }) async {
    final formData = FormData.fromMap({
      'file': await MultipartFile.fromFile(
        filePath,
        filename: fileName,
      ),
    });

    final response = await _dio.post<Map<String, dynamic>>(
      ApiConstants.filesUpload,
      data: formData,
      options: Options(
        headers: {'Content-Type': 'multipart/form-data'},
      ),
      onSendProgress: onProgress,
    );

    final result = response.data!;
    if (result['code']?.toString() == ApiConstants.successCode) {
      return FileUploadResponse.fromJson(
        result['data'] as Map<String, dynamic>,
      );
    }
    throw Exception(result['msg'] ?? '文件上传失败');
  }

  /// MD5 秒传检查
  ///
  /// GET /files/check?md5={md5}
  /// 返回 true 表示文件已存在，可跳过上传
  Future<bool> checkMd5(String md5) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.filesCheck,
      queryParameters: {'md5': md5},
    );

    final result = response.data!;
    if (result['code']?.toString() == ApiConstants.successCode) {
      return result['data'] as bool? ?? false;
    }
    return false;
  }

  /// 获取文件详情
  ///
  /// GET /files/{fileId}
  Future<FileModel> getFileDetail(String fileId) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.files}/$fileId',
    );

    final result = response.data!;
    if (result['code']?.toString() == ApiConstants.successCode) {
      return FileModel.fromJson(result['data'] as Map<String, dynamic>);
    }
    throw Exception(result['msg'] ?? '获取文件信息失败');
  }

  /// 删除文件
  ///
  /// DELETE /files/{fileId}
  Future<void> deleteFile(String fileId) async {
    await _dio.delete<Map<String, dynamic>>(
      '${ApiConstants.files}/$fileId',
    );
  }
}
