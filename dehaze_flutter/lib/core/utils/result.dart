import '../errors/failures.dart';

class Result<T> {
  final T? data;
  final Failure? error;
  final bool isSuccess;

  const Result._({
    this.data,
    this.error,
    required this.isSuccess,
  });

  factory Result.success(T data) {
    return Result._(data: data, isSuccess: true);
  }

  factory Result.failure(Failure error) {
    return Result._(error: error, isSuccess: false);
  }

  bool get isFailure => !isSuccess;

  T get dataOrThrow {
    if (isSuccess && data != null) {
      return data!;
    } else {
      throw error ?? const UnknownFailure('No data available');
    }
  }

  Result<R> map<R>(R Function(T data) successMapper) {
    if (isSuccess && data != null) {
      try {
        return Result.success(successMapper(data as T));
      } catch (e) {
        return Result.failure(UnknownFailure('Mapping failed', e));
      }
    } else {
      return Result.failure(error ?? const UnknownFailure('No data available'));
    }
  }

  Result<R> flatMap<R>(Result<R> Function(T data) successMapper) {
    if (isSuccess && data != null) {
      try {
        return successMapper(data as T);
      } catch (e) {
        return Result.failure(UnknownFailure('Flat mapping failed', e));
      }
    } else {
      return Result.failure(error ?? const UnknownFailure('No data available'));
    }
  }

  void fold(
    void Function(Failure error) onFailure,
    void Function(T data) onSuccess,
  ) {
    if (isSuccess && data != null) {
      onSuccess(data as T);
    } else {
      onFailure(error ?? const UnknownFailure('No data available'));
    }
  }

  T? getOrNull() => isSuccess ? data : null;

  T? get dataOrNull => isSuccess ? data : null;

  Failure? getErrorOrNull() => isFailure ? error : null;

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is Result<T> &&
        other.data == data &&
        other.error == error &&
        other.isSuccess == isSuccess;
  }

  @override
  int get hashCode => data.hashCode ^ error.hashCode ^ isSuccess.hashCode;

  @override
  String toString() {
    return isSuccess ? 'Result.success($data)' : 'Result.failure($error)';
  }
}