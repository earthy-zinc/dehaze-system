import '../entities/dehaze_image.dart';
import '../../../../core/utils/result.dart';

abstract class DehazeRepository {
  Future<Result<List<DehazeImage>>> getDehazeHistory();
  Future<Result<DehazeImage>> processImage(
    String imagePath,
    DehazeParameters parameters,
  );
  Future<Result<void>> saveDehazeImage(DehazeImage image);
  Future<Result<void>> deleteDehazeImage(String imageId);
  Future<Result<DehazeImage>> getDehazeImageById(String imageId);
  Future<Result<List<DehazeAlgorithm>>> getAvailableAlgorithms();
  Future<Result<void>> cancelProcessing(String imageId);
  Stream<Result<DehazeImage>> watchProcessingStatus(String imageId);
}