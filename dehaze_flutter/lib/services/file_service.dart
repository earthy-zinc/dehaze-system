import 'dart:typed_data';

import 'package:dio/dio.dart';

import '../core/constants/api_constants.dart';
import '../models/file_model.dart';

/// 文件服务
///
/// 封装文件上传 API（POST /files，multipart/form-data）。
/// 本地图片在选择时即被读取为字节流，直接上传，
/// 避免依赖 dart:io 的文件路径（Web 端不可用）。
class FileService {
  const FileService(this._dio);

  final Dio _dio;

  /// 从字节流上传文件
  Future<FileUploadResponse> uploadBytes(
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
      ApiConstants.filesUpload,
      data: formData,
      onSendProgress: onProgress,
    );
    // ResponseInterceptor 已保证 code=='00000'，失败已 reject 为 ApiException
    return FileUploadResponse.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }
}
