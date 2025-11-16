import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/dehaze_image_model.dart';
import '../../../../core/utils/result.dart';
import '../../../../core/errors/failures.dart';

abstract class DehazeLocalDataSource {
  Future<Result<List<DehazeImageModel>>> getDehazeHistory();
  Future<Result<void>> saveDehazeImage(DehazeImageModel image);
  Future<Result<void>> deleteDehazeImage(String imageId);
  Future<Result<DehazeImageModel?>> getDehazeImageById(String imageId);
  Future<Result<void>> clearAll();
}

class DehazeLocalDataSourceImpl implements DehazeLocalDataSource {
  final SharedPreferences sharedPreferences;
  static const String _dehazeHistoryKey = 'dehaze_history';
  static const int _maxHistoryCount = 50;

  DehazeLocalDataSourceImpl(this.sharedPreferences);

  @override
  Future<Result<List<DehazeImageModel>>> getDehazeHistory() async {
    try {
      final historyJson = sharedPreferences.getStringList(_dehazeHistoryKey);
      if (historyJson == null || historyJson.isEmpty) {
        return Result.success([]);
      }

      final images = historyJson
          .map((jsonString) => DehazeImageModel.fromJson(jsonDecode(jsonString)))
          .toList();

      images.sort((a, b) => b.createdAt.compareTo(a.createdAt));

      return Result.success(images);
    } catch (e) {
      return Result.failure(CacheFailure('Failed to load dehaze history: $e'));
    }
  }

  @override
  Future<Result<void>> saveDehazeImage(DehazeImageModel image) async {
    try {
      final result = await getDehazeHistory();

      if (result.isFailure) {
        final images = [image];
        await _saveImagesList(images);
      } else {
        final existingImages = result.dataOrNull ?? [];
        final index = existingImages.indexWhere((img) => img.id == image.id);
        final images = List<DehazeImageModel>.from(existingImages);

        if (index >= 0) {
          images[index] = image;
        } else {
          images.insert(0, image);
          if (images.length > _maxHistoryCount) {
            images.removeRange(_maxHistoryCount, images.length);
          }
        }

        await _saveImagesList(images);
      }

      return Result.success(null);
    } catch (e) {
      return Result.failure(CacheFailure('Failed to save dehaze image: $e'));
    }
  }

  @override
  Future<Result<void>> deleteDehazeImage(String imageId) async {
    try {
      final result = await getDehazeHistory();

      if (result.isFailure) {
        return Result.failure(result.getErrorOrNull()!);
      }

      final images = result.dataOrNull ?? [];
      final filteredImages = images.where((img) => img.id != imageId).toList();
      await _saveImagesList(filteredImages);
      return Result.success(null);
    } catch (e) {
      return Result.failure(CacheFailure('Failed to delete dehaze image: $e'));
    }
  }

  @override
  Future<Result<DehazeImageModel?>> getDehazeImageById(String imageId) async {
    try {
      final result = await getDehazeHistory();

      if (result.isFailure) {
        return Result.failure(result.getErrorOrNull()!);
      }

      final images = result.dataOrNull ?? [];
      final image = images.cast<DehazeImageModel?>().firstWhere(
        (img) => img?.id == imageId,
        orElse: () => null,
      );
      return Result.success(image);
    } catch (e) {
      return Result.failure(CacheFailure('Failed to get dehaze image by ID: $e'));
    }
  }

  @override
  Future<Result<void>> clearAll() async {
    try {
      await sharedPreferences.remove(_dehazeHistoryKey);
      return Result.success(null);
    } catch (e) {
      return Result.failure(CacheFailure('Failed to clear dehaze history: $e'));
    }
  }

  Future<void> _saveImagesList(List<DehazeImageModel> images) async {
    final imagesJson = images
        .map((image) => jsonEncode(image.toJson()))
        .toList();

    await sharedPreferences.setStringList(_dehazeHistoryKey, imagesJson);
  }
}