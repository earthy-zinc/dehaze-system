class NetworkException implements Exception {

  NetworkException({required this.message, this.statusCode, this.response});

  factory NetworkException.fromDioError(dynamic error) {
    if (error is! Map<String, dynamic>) {
      return NetworkException(
        message: 'Unknown network error occurred.',
      );
    }

    final errorType = error['type'] as Map<String, dynamic>?;
    final typeName = errorType?['name'] as String?;

    if (typeName == 'connectTimeout') {
      return NetworkException(
        message: 'Connection timeout. Please check your internet connection.',
        statusCode: 408,
      );
    }

    if (typeName == 'receiveTimeout') {
      return NetworkException(
        message: 'Receive timeout. Please try again.',
        statusCode: 408,
      );
    }

    if (typeName == 'sendTimeout') {
      return NetworkException(
        message: 'Send timeout. Please try again.',
        statusCode: 408,
      );
    }

    if (typeName == 'cancel') {
      return NetworkException(message: 'Request was cancelled.');
    }

    if (typeName == 'unknown') {
      return NetworkException(
        message: 'No internet connection.',
        statusCode: 0,
      );
    }

    final response = error['response'] as Map<String, dynamic>?;
    if (response != null) {
      final statusCode = response['statusCode'] as int?;
      final message = _getErrorMessage(statusCode, response['data']);

      return NetworkException(
        message: message,
        statusCode: statusCode,
        response: response['data']?.toString(),
      );
    }

    return NetworkException(
      message: error['message'] as String? ?? 'Unknown network error occurred.',
      statusCode: response?['statusCode'] as int?,
    );
  }
  final String message;
  final int? statusCode;
  final String? response;

  @override
  String toString() => 'NetworkException: $message${statusCode != null ? ' (Status: $statusCode)' : ''}';

  static String _getErrorMessage(int? statusCode, dynamic responseData) {
    switch (statusCode) {
      case 400:
        return (responseData as Map<String, dynamic>?)?['message'] as String? ??
            'Bad request. Please check your input.';
      case 401:
        return 'Unauthorized. Please login again.';
      case 403:
        return 'Forbidden. You don\'t have permission to access this resource.';
      case 404:
        return 'Not found. The requested resource was not found.';
      case 405:
        return 'Method not allowed.';
      case 408:
        return 'Request timeout. Please try again.';
      case 429:
        return 'Too many requests. Please try again later.';
      case 500:
        return 'Internal server error. Please try again later.';
      case 502:
        return 'Bad gateway. The server is temporarily unavailable.';
      case 503:
        return 'Service unavailable. Please try again later.';
      case 504:
        return 'Gateway timeout. Please try again later.';
      default:
        if (responseData is Map<String, dynamic>) {
          return responseData['message'] as String? ??
              'Something went wrong. Please try again.';
        }
        return 'Something went wrong. Please try again.';
    }
  }
}
