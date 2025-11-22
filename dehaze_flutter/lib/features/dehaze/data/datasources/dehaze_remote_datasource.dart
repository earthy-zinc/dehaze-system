import 'dart:io';

import '../../../../core/errors/failures.dart';
import '../../../../core/network/api_config.dart';
import '../../../../core/network/dio_client.dart';
import '../../../../core/network/network_exceptions.dart';
import '../../../../core/utils/result.dart';
import '../../domain/entities/dehaze_image.dart';
import '../models/dehaze_image_model.dart';

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

  DehazeRemoteDataSourceImpl(this.dioClient);
  final DioClient dioClient;

  @override
  Future<Result<DehazeImageModel>> processImage(
    String imagePath,
    DehazeParameters parameters,
  ) async {
    try {
      final file = File(imagePath);
      if (!file.existsSync()) {
        return Result.failure(
          const ValidationFailure('Image file does not exist'),
        );
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
        final imageModel = DehazeImageModel.fromJson(response['data'] as Map<String, dynamic>);
        return Result.success(imageModel);
      } else {
        return Result.failure(
          ServerFailure('Failed to process image: ${response['statusCode']}'),
        );
      }
    } on NetworkException catch (e) {
      return Result.failure(NetworkFailure(e.message));
    } on Exception catch (e) {
      return Result.failure(NetworkFailure('Unexpected error: $e'));
    }
  }

  @override
  Future<Result<List<DehazeAlgorithm>>> getAvailableAlgorithms() async {
    try {
      final response = await dioClient.get(ApiEndpoints.getAlgorithms);

      if (response['statusCode'] == HttpStatusCodes.ok) {
        final responseData = response['data'] as Map<String, dynamic>;
        final algorithmsJson = responseData['algorithms'] as List;
        final algorithms = algorithmsJson
            .map((json) => _parseAlgorithm(json as String))
            .toList();
        return Result.success(algorithms);
      } else {
        return Result.failure(
          ServerFailure('Failed to get algorithms: ${response['statusCode']}'),
        );
      }
    } on Exception catch (e) {
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
        return Result.failure(
          ServerFailure(
            'Failed to cancel processing: ${response['statusCode']}',
          ),
        );
      }
    } on Exception catch (e) {
      return Result.failure(NetworkFailure('Network error: $e'));
    }
  }

  @override
  Stream<Result<DehazeImageModel>> watchProcessingStatus(
    String imageId,
  ) async* {
    try {
      while (true) {
        await Future<void>.delayed(const Duration(seconds: 2));

        final response = await dioClient.get('/api/dehaze/status/$imageId');

        if (response['statusCode'] == 200) {
          final imageModel = DehazeImageModel.fromJson(response['data'] as Map<String, dynamic>);
          yield Result.success(imageModel);

          if (imageModel.status == ProcessingStatus.completed ||
              imageModel.status == ProcessingStatus.failed ||
              imageModel.status == ProcessingStatus.cancelled) {
            break;
          }
        } else {
          yield Result.failure(
            ServerFailure('Failed to get status: ${response['statusCode']}'),
          );
          break;
        }
      }
    } on Exception catch (e) {
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
