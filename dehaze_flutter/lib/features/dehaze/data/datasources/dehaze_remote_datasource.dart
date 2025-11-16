import 'dart:io';
import '../models/dehaze_image_model.dart';
import '../../domain/entities/dehaze_image.dart';
import '../../../../core/utils/result.dart';
import '../../../../core/errors/failures.dart';
import '../../../../core/network/dio_client.dart';
import '../../../../core/network/api_config.dart';
import '../../../../core/network/network_exceptions.dart';

abstract class DehazeRemoteDataSource {
  Future<Result<DehazeImageModel>> processImage(
    String imagePath,
    DehazeParameters parameters,
  );
  Future<Result<List<DehazeAlgorithm>>> getAvailableAlgorithms();
  Future<Result<void>> cancelProcessing(String imageId);
  Stream<Result<DehazeImageModel>> watchProcessingStatus(String imageId);
}

class DehazeRemoteDataSourceImpl implements DehazeRemoteDataSource {
  final DioClient dioClient;

  DehazeRemoteDataSourceImpl(this.dioClient);

  @override
  Future<Result<DehazeImageModel>> processImage(
    String imagePath,
    DehazeParameters parameters,
  ) async {
    try {
      final file = File(imagePath);
      if (!file.existsSync()) {
        return Result.failure(const ValidationFailure('Image file does not exist'));
      }

      final data = {
        'parameters': DehazeParametersModel.fromEntity(parameters).toJson(),
      };

      final response = await dioClient.post(
        ApiEndpoints.dehaze,
        data: data,
        file: file,
      );

      if (response['statusCode'] == HttpStatusCodes.ok ||
          response['statusCode'] == HttpStatusCodes.created) {
        final imageModel = DehazeImageModel.fromJson(response['data']);
        return Result.success(imageModel);
      } else {
        return Result.failure(ServerFailure(
          'Failed to process image: ${response['statusCode']}',
        ));
      }
    } on NetworkException catch (e) {
      return Result.failure(NetworkFailure(e.message));
    } catch (e) {
      return Result.failure(NetworkFailure('Unexpected error: $e'));
    }
  }

  @override
  Future<Result<List<DehazeAlgorithm>>> getAvailableAlgorithms() async {
    try {
      final response = await dioClient.get(ApiEndpoints.getAlgorithms);

      if (response['statusCode'] == HttpStatusCodes.ok) {
        final algorithmsJson = response['data']['algorithms'] as List;
        final algorithms = algorithmsJson
            .map((json) => _parseAlgorithm(json as String))
            .toList();
        return Result.success(algorithms);
      } else {
        return Result.failure(ServerFailure(
          'Failed to get algorithms: ${response['statusCode']}',
        ));
      }
    } catch (e) {
      return Result.failure(NetworkFailure('Network error: $e'));
    }
  }

  @override
  Future<Result<void>> cancelProcessing(String imageId) async {
    try {
      final response = await dioClient.post('/api/dehaze/cancel/$imageId');

      if (response['statusCode'] == 200) {
        return Result.success(null);
      } else {
        return Result.failure(ServerFailure(
          'Failed to cancel processing: ${response['statusCode']}',
        ));
      }
    } catch (e) {
      return Result.failure(NetworkFailure('Network error: $e'));
    }
  }

  @override
  Stream<Result<DehazeImageModel>> watchProcessingStatus(String imageId) async* {
    try {
      while (true) {
        await Future.delayed(const Duration(seconds: 2));

        final response = await dioClient.get('/api/dehaze/status/$imageId');

        if (response['statusCode'] == 200) {
          final imageModel = DehazeImageModel.fromJson(response['data']);
          yield Result.success(imageModel);

          if (imageModel.status == ProcessingStatus.completed ||
              imageModel.status == ProcessingStatus.failed ||
              imageModel.status == ProcessingStatus.cancelled) {
            break;
          }
        } else {
          yield Result.failure(ServerFailure(
            'Failed to get status: ${response['statusCode']}',
          ));
          break;
        }
      }
    } catch (e) {
      yield Result.failure(NetworkFailure('Network error: $e'));
    }
  }

  DehazeAlgorithm _parseAlgorithm(String algorithmString) {
    switch (algorithmString.toLowerCase()) {
      case 'darkchannel':
        return DehazeAlgorithm.darkChannel;
      case 'atmosphericlight':
        return DehazeAlgorithm.atmosphericLight;
      case 'retinex':
        return DehazeAlgorithm.retinex;
      case 'colorattenuation':
        return DehazeAlgorithm.colorAttenuation;
      case 'custom':
        return DehazeAlgorithm.custom;
      default:
        return DehazeAlgorithm.darkChannel;
    }
  }

  }