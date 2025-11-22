abstract class Failure implements Exception {

  const Failure(this.message);
  final String message;

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) {
      return true;
    }
    return other is Failure && other.message == message;
  }

  @override
  int get hashCode => message.hashCode;

  @override
  String toString() => 'Failure(message: $message)';
}

class ServerFailure extends Failure {
  const ServerFailure(super.message);
}

class NetworkFailure extends Failure {
  const NetworkFailure(super.message);
}

class CacheFailure extends Failure {
  const CacheFailure(super.message);
}

class ValidationFailure extends Failure {
  const ValidationFailure(super.message);
}

class ProcessingFailure extends Failure {

  const ProcessingFailure(super.message, [this.errorCode]);
  final String? errorCode;

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) {
      return true;
    }
    return other is ProcessingFailure &&
        other.message == message &&
        other.errorCode == errorCode;
  }

  @override
  int get hashCode => message.hashCode ^ errorCode.hashCode;
}

class StorageFailure extends Failure {
  const StorageFailure(super.message);
}

class PermissionFailure extends Failure {
  const PermissionFailure(super.message);
}

class UnknownFailure extends Failure implements Exception {

  const UnknownFailure(super.message, [this.originalError]);
  final dynamic originalError;

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) {
      return true;
    }
    return other is UnknownFailure &&
        other.message == message &&
        other.originalError == originalError;
  }

  @override
  int get hashCode => message.hashCode ^ originalError.hashCode;
}
