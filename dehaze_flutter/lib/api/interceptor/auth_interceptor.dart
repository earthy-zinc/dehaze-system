import 'package:dio/dio.dart';

class AuthInterceptor extends Interceptor {
  @override
  Future<void> onError(
    DioException err,
    ErrorInterceptorHandler handler,
  ) async {
    super.onError(err, handler);
  }
}
