import 'dart:io';
import 'package:dio/dio.dart';
import 'package:logging/logging.dart';
import 'api_config.dart';
import 'network_exceptions.dart';

abstract class DioClient {
  Future<Map<String, dynamic>> get(String path);
  Future<Map<String, dynamic>> post(String path, {Map<String, dynamic>? data, File? file});
  Future<Map<String, dynamic>> put(String path, {Map<String, dynamic>? data});
  Future<Map<String, dynamic>> delete(String path);
  Future<void> setAuthToken(String token);
  Future<void> clearAuthToken();
}

class DioClientImpl implements DioClient {
  final Dio _dio;
  static final _logger = Logger('DioClient');

  DioClientImpl() : _dio = _createDioInstance();

  static Dio _createDioInstance() {
    final dio = Dio(BaseOptions(
      baseUrl: ApiConfig.buildUrl(''),
      connectTimeout: ApiConfig.connectTimeout,
      receiveTimeout: ApiConfig.receiveTimeout,
      sendTimeout: ApiConfig.sendTimeout,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
    ));

    // 添加请求/响应拦截器
    dio.interceptors.add(
      InterceptorsWrapper(
        onRequest: (options, handler) {
          // 添加请求日志（仅在调试模式下）
          if (!const bool.fromEnvironment('dart.vm.product')) {
            _logger.info('🌐 Request: ${options.method} ${options.uri}');
            if (options.data != null) {
              _logger.info('📤 Data: ${options.data}');
            }
          }
          handler.next(options);
        },
        onResponse: (response, handler) {
          // 添加响应日志（仅在调试模式下）
          if (!const bool.fromEnvironment('dart.vm.product')) {
            _logger.info('🌐 Response: ${response.statusCode} ${response.requestOptions.uri}');
            _logger.info('📥 Data: ${response.data}');
          }
          handler.next(response);
        },
        onError: (error, handler) {
          // 添加错误日志（仅在调试模式下）
          if (!const bool.fromEnvironment('dart.vm.product')) {
            _logger.severe('❌ Error: ${error.message}');
            if (error.response?.data != null) {
              _logger.severe('📥 Error Data: ${error.response?.data}');
            }
          }
          handler.next(error);
        },
      ),
    );

    return dio;
  }

  @override
  Future<Map<String, dynamic>> get(String path) async {
    try {
      final response = await _dio.get(path);
      return _handleResponse(response);
    } on DioException catch (e) {
      throw NetworkException.fromDioError(e);
    } catch (e) {
      throw NetworkException(message: 'Unexpected error: $e');
    }
  }

  @override
  Future<Map<String, dynamic>> post(String path, {Map<String, dynamic>? data, File? file}) async {
    try {
      dynamic requestData;

      if (file != null) {
        // 如果有文件上传，使用FormData
        requestData = FormData.fromMap({
          'file': await MultipartFile.fromFile(file.path),
          ...?data,
        });
      } else {
        requestData = data;
      }

      final response = await _dio.post(
        path,
        data: requestData,
        options: file != null
          ? Options(contentType: 'multipart/form-data')
          : null,
      );

      return _handleResponse(response);
    } on DioException catch (e) {
      throw NetworkException.fromDioError(e);
    } catch (e) {
      throw NetworkException(message: 'Unexpected error: $e');
    }
  }

  @override
  Future<Map<String, dynamic>> put(String path, {Map<String, dynamic>? data}) async {
    try {
      final response = await _dio.put(path, data: data);
      return _handleResponse(response);
    } on DioException catch (e) {
      throw NetworkException.fromDioError(e);
    } catch (e) {
      throw NetworkException(message: 'Unexpected error: $e');
    }
  }

  @override
  Future<Map<String, dynamic>> delete(String path) async {
    try {
      final response = await _dio.delete(path);
      return _handleResponse(response);
    } on DioException catch (e) {
      throw NetworkException.fromDioError(e);
    } catch (e) {
      throw NetworkException(message: 'Unexpected error: $e');
    }
  }

  @override
  Future<void> setAuthToken(String token) async {
    _dio.options.headers['Authorization'] = 'Bearer $token';
  }

  @override
  Future<void> clearAuthToken() async {
    _dio.options.headers.remove('Authorization');
  }

  Map<String, dynamic> _handleResponse(Response response) {
    switch (response.statusCode) {
      case HttpStatusCodes.ok:
      case HttpStatusCodes.created:
        return {
          'statusCode': response.statusCode,
          'data': response.data is Map<String, dynamic>
            ? response.data as Map<String, dynamic>
            : {'response': response.data},
        };
      case HttpStatusCodes.noContent:
        return {
          'statusCode': response.statusCode,
          'data': {},
        };
      default:
        throw NetworkException(
          message: 'HTTP ${response.statusCode}: ${response.statusMessage}',
          statusCode: response.statusCode,
          response: response.data?.toString(),
        );
    }
  }

  // 获取Dio实例（用于特殊情况）
  Dio get dioInstance => _dio;
}