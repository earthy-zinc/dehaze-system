import '../errors/failures.dart';

class Result<T> {

  const Result._({required this.isSuccess, this.data, this.error});

  factory Result.success(T data) => Result._(data: data, isSuccess: true);

  factory Result.failure(Failure error) => Result._(error: error, isSuccess: false);
  final T? data;
  final Failure? error;
  final bool isSuccess;

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
      } on Exception {
        return Result.failure(const UnknownFailure('Mapping failed'));
      }
    } else {
      return Result.failure(error ?? const UnknownFailure('No data available'));
    }
  }

  Result<R> flatMap<R>(Result<R> Function(T data) successMapper) {
    if (isSuccess && data != null) {
      try {
        return successMapper(data as T);
      } on Exception {
        return Result.failure(const UnknownFailure('Flat mapping failed'));
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
    if (identical(this, other)) {
      return true;
    }
    return other is Result<T> &&
        other.data == data &&
        other.error == error &&
        other.isSuccess == isSuccess;
  }

  @override
  int get hashCode => data.hashCode ^ error.hashCode ^ isSuccess.hashCode;

  @override
  String toString() => isSuccess ? 'Result.success($data)' : 'Result.failure($error)';
}
