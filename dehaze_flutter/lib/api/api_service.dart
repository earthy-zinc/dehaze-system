import 'package:dio/dio.dart';
import 'api_config.dart';
import 'interceptor/auth_interceptor.dart';

class APIService {
  factory APIService() => _instance;

  APIService._internal();

  static final APIService _instance = APIService._internal();

  late final Dio _dio;

  void initialize() {
    _dio = Dio(
      BaseOptions(
        baseUrl: ApiConfig.buildUrl(''),
        connectTimeout: ApiConfig.connectTimeout,
        receiveTimeout: ApiConfig.receiveTimeout,
        sendTimeout: ApiConfig.sendTimeout,
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
      ),
    );

    // 添加请求/响应拦截器
    _dio.interceptors.addAll([AuthInterceptor()]);
  }

  Dio get dio => _dio;
}
