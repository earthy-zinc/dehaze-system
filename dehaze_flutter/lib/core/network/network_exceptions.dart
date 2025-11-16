class NetworkException implements Exception {
  final String message;
  final int? statusCode;
  final String? response;

  NetworkException({
    required this.message,
    this.statusCode,
    this.response,
  });

  @override
  String toString() {
    return 'NetworkException: $message${statusCode != null ? ' (Status: $statusCode)' : ''}';
  }

  factory NetworkException.fromDioError(dynamic error) {
    if (error.type?.name == 'connectTimeout') {
      return NetworkException(
        message: 'Connection timeout. Please check your internet connection.',
        statusCode: 408,
      );
    }

    if (error.type?.name == 'receiveTimeout') {
      return NetworkException(
        message: 'Receive timeout. Please try again.',
        statusCode: 408,
      );
    }

    if (error.type?.name == 'sendTimeout') {
      return NetworkException(
        message: 'Send timeout. Please try again.',
        statusCode: 408,
      );
    }

    if (error.type?.name == 'cancel') {
      return NetworkException(
        message: 'Request was cancelled.',
      );
    }

    if (error.type?.name == 'unknown') {
      return NetworkException(
        message: 'No internet connection.',
        statusCode: 0,
      );
    }

    if (error.response != null) {
      final statusCode = error.response?.statusCode;
      final message = _getErrorMessage(statusCode, error.response?.data);

      return NetworkException(
        message: message,
        statusCode: statusCode,
        response: error.response?.data?.toString(),
      );
    }

    return NetworkException(
      message: error.message ?? 'Unknown network error occurred.',
      statusCode: error.response?.statusCode,
    );
  }

  static String _getErrorMessage(int? statusCode, dynamic responseData) {
    switch (statusCode) {
      case 400:
        return responseData?['message'] ?? 'Bad request. Please check your input.';
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
        return responseData?['message'] ?? 'Something went wrong. Please try again.';
    }
  }
}