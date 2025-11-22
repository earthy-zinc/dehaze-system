import '../../../../core/errors/failures.dart';
import '../../../../core/network/network_info.dart';
import '../../../../core/utils/result.dart';
import '../../domain/entities/dehaze_image.dart';
import '../../domain/repositories/dehaze_repository.dart';
import '../datasources/dehaze_local_datasource.dart';
import '../datasources/dehaze_remote_datasource.dart';
import '../models/dehaze_image_model.dart';

class DehazeRepositoryImpl implements DehazeRepository {

  DehazeRepositoryImpl({
    required this.localDataSource,
    required this.remoteDataSource,
    required this.networkInfo,
  });
  final DehazeLocalDataSource localDataSource;
  final DehazeRemoteDataSource remoteDataSource;
  final NetworkInfo networkInfo;

  @override
  Future<Result<List<DehazeImage>>> getDehazeHistory() async {
    try {
      final result = await localDataSource.getDehazeHistory();
      return result.map(
        (images) => images.map((model) => model.toEntity()).toList(),
      );
    } on Exception catch (e) {
      return Result.failure(CacheFailure('Failed to get dehaze history: $e'));
    }
  }

  @override
  Future<Result<DehazeImage>> processImage(
    String imagePath,
    DehazeParameters parameters,
  ) async {
    if (await networkInfo.isConnected) {
      try {
        final result = await remoteDataSource.processImage(
          imagePath,
          parameters,
        );

        if (result.isSuccess) {
          final imageModel = result.dataOrThrow;
          await localDataSource.saveDehazeImage(imageModel);
          return Result.success(imageModel.toEntity());
        } else {
          return Result.failure(result.getErrorOrNull()!);
        }
      } on Exception catch (e) {
        return Result.failure(
          NetworkFailure('Failed to process image remotely: $e'),
        );
      }
    } else {
      return Result.failure(const NetworkFailure('No internet connection'));
    }
  }

  @override
  Future<Result<void>> saveDehazeImage(DehazeImage image) async {
    try {
      final imageModel = DehazeImageModel.fromEntity(image);
      return await localDataSource.saveDehazeImage(imageModel);
    } on Exception catch (e) {
      return Result.failure(CacheFailure('Failed to save dehaze image: $e'));
    }
  }

  @override
  Future<Result<void>> deleteDehazeImage(String imageId) async {
    try {
      final result = await localDataSource.deleteDehazeImage(imageId);

      if (result.isSuccess && await networkInfo.isConnected) {
        await remoteDataSource.cancelProcessing(imageId);
      }

      return result;
    } on Exception catch (e) {
      return Result.failure(CacheFailure('Failed to delete dehaze image: $e'));
    }
  }

  @override
  Future<Result<DehazeImage>> getDehazeImageById(String imageId) async {
    try {
      final result = await localDataSource.getDehazeImageById(imageId);

      return result.map((model) {
        if (model != null) {
          return model.toEntity();
        } else {
          throw Exception('Image not found');
        }
      });
    } on Exception catch (e) {
      return Result.failure(
        CacheFailure('Failed to get dehaze image by ID: $e'),
      );
    }
  }

  @override
  Future<Result<List<DehazeAlgorithm>>> getAvailableAlgorithms() async {
    if (await networkInfo.isConnected) {
      try {
        return await remoteDataSource.getAvailableAlgorithms();
      } on Exception catch (e) {
        return Result.failure(
          NetworkFailure('Failed to get algorithms remotely: $e'),
        );
      }
    } else {
      return Result.success([
        DehazeAlgorithm.darkChannel,
        DehazeAlgorithm.atmosphericLight,
        DehazeAlgorithm.retinex,
        DehazeAlgorithm.colorAttenuation,
      ]);
    }
  }

  @override
  Future<Result<void>> cancelProcessing(String imageId) async {
    try {
      final result = await localDataSource.deleteDehazeImage(imageId);

      if (result.isSuccess && await networkInfo.isConnected) {
        await remoteDataSource.cancelProcessing(imageId);
      }

      return result;
    } on Exception catch (e) {
      return Result.failure(CacheFailure('Failed to cancel processing: $e'));
    }
  }

  @override
  Stream<Result<DehazeImage>> watchProcessingStatus(String imageId) async* {
    if (await networkInfo.isConnected) {
      yield* remoteDataSource
          .watchProcessingStatus(imageId)
          .map((model) => model.map((m) => m.toEntity()));
    } else {
      yield Result.failure(const NetworkFailure('No internet connection'));
    }
  }
}
