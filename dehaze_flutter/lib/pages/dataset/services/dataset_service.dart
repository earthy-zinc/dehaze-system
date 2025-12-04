import 'package:dio/dio.dart';
import '../models/dataset_model.dart';

class DatasetService {
  const DatasetService(this._dio);

  final Dio _dio;

  // 获取数据集列表
  Future<PaginatedDatasetResponse> fetchDatasets({
    int page = 1,
    int pageSize = 10,
    String search = '',
  }) async {
    try {
      final response = await _dio.get(
        '/datasets',
        queryParameters: {
          'page': page,
          'page_size': pageSize,
          if (search.isNotEmpty) 'search': search,
        },
      );

      if (response.statusCode == 200 && response.data['code'] == 0) {
        final data = response.data['data'] as Map<String, dynamic>;
        return PaginatedDatasetResponse.fromJson(data);
      }
      throw Exception('Failed to fetch datasets: ${response.statusCode}');
    } on DioException catch (e) {
      // Mock数据用于开发阶段
      if (e.type == DioExceptionType.connectionError ||
          e.type == DioExceptionType.connectionTimeout) {
        return _getMockDatasets(page, search);
      }
      rethrow;
    }
  }

  // 获取数据集详情
  Future<DatasetModel> fetchDatasetDetail(int datasetId) async {
    try {
      final response = await _dio.get('/datasets/$datasetId');

      if (response.statusCode == 200 && response.data['code'] == 0) {
        final data = response.data['data'] as Map<String, dynamic>;
        return DatasetModel.fromJson(data);
      }
      throw Exception('Failed to fetch dataset detail: ${response.statusCode}');
    } on DioException catch (e) {
      if (e.type == DioExceptionType.connectionError ||
          e.type == DioExceptionType.connectionTimeout) {
        return _getMockDatasetDetail(datasetId);
      }
      rethrow;
    }
  }

  // 获取数据集图片列表
  Future<PaginatedImageResponse> fetchDatasetImages({
    required int datasetId,
    int page = 1,
    int pageSize = 20,
    ImageType? imageType,
    String search = '',
  }) async {
    try {
      final response = await _dio.get(
        '/datasets/$datasetId/images',
        queryParameters: {
          'page': page,
          'page_size': pageSize,
          if (imageType != null) 'image_type': imageType.name,
          if (search.isNotEmpty) 'search': search,
        },
      );

      if (response.statusCode == 200 && response.data['code'] == 0) {
        final data = response.data['data'] as Map<String, dynamic>;
        return PaginatedImageResponse.fromJson(data);
      }
      throw Exception('Failed to fetch dataset images: ${response.statusCode}');
    } on DioException catch (e) {
      if (e.type == DioExceptionType.connectionError ||
          e.type == DioExceptionType.connectionTimeout) {
        return _getMockDatasetImages(datasetId, page, imageType, search);
      }
      rethrow;
    }
  }

  // Mock数据方法
  PaginatedDatasetResponse _getMockDatasets(int page, String search) {
    final allDatasets = [
      DatasetModel(
        id: 1,
        name: 'RESIDE数据集',
        description: '大规模真实场景图像去雾数据集，包含室内外多种场景',
        creator: 'Li Boyi',
        thumbnail:
            'https://images.unsplash.com/photo-1500534314209-a25ddb2bd429?w=400&h=400&fit=crop',
        totalImages: 13990,
        foggyCount: 6995,
        clearCount: 6995,
        annotatedCount: 0,
        createdAt: DateTime.now().subtract(const Duration(days: 15)),
        updatedAt: DateTime.now().subtract(const Duration(days: 15)),
      ),
      DatasetModel(
        id: 2,
        name: 'O-HAZE数据集',
        description: '户外真实雾霾图像数据集，包含45对有雾/无雾图像',
        creator: 'Ancuti Codruta',
        thumbnail:
            'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&h=400&fit=crop',
        totalImages: 90,
        foggyCount: 45,
        clearCount: 45,
        annotatedCount: 0,
        createdAt: DateTime.now().subtract(const Duration(days: 10)),
        updatedAt: DateTime.now().subtract(const Duration(days: 10)),
      ),
      DatasetModel(
        id: 3,
        name: 'I-HAZE数据集',
        description: '室内真实雾霾图像数据集，包含35对有雾/无雾图像',
        creator: 'Ancuti Codruta',
        thumbnail:
            'https://images.unsplash.com/photo-1497366216548-37526070297c?w=400&h=400&fit=crop',
        totalImages: 70,
        foggyCount: 35,
        clearCount: 35,
        annotatedCount: 0,
        createdAt: DateTime.now().subtract(const Duration(days: 8)),
        updatedAt: DateTime.now().subtract(const Duration(days: 8)),
      ),
      DatasetModel(
        id: 4,
        name: 'Dense-Haze数据集',
        description: '密集雾霾场景数据集，专注于极端雾霾条件',
        creator: 'Ancuti Codruta',
        thumbnail:
            'https://images.unsplash.com/photo-1519681393784-d120267933ba?w=400&h=400&fit=crop',
        totalImages: 110,
        foggyCount: 55,
        clearCount: 55,
        annotatedCount: 0,
        createdAt: DateTime.now().subtract(const Duration(days: 5)),
        updatedAt: DateTime.now().subtract(const Duration(days: 5)),
      ),
      DatasetModel(
        id: 5,
        name: 'NH-HAZE数据集',
        description: '非均匀雾霾数据集，模拟真实世界的复杂雾霾分布',
        creator: 'Ancuti Codruta',
        thumbnail:
            'https://images.unsplash.com/photo-1519681393784-d120267933ba?w=400&h=400&fit=crop',
        totalImages: 110,
        foggyCount: 55,
        clearCount: 55,
        annotatedCount: 0,
        createdAt: DateTime.now().subtract(const Duration(days: 3)),
        updatedAt: DateTime.now().subtract(const Duration(days: 3)),
      ),
      DatasetModel(
        id: 6,
        name: 'SOTS数据集',
        description: '合成雾霾数据集，包含室内外场景',
        creator: 'Li Boyi',
        thumbnail:
            'https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?w=400&h=400&fit=crop',
        totalImages: 1000,
        foggyCount: 500,
        clearCount: 500,
        annotatedCount: 0,
        createdAt: DateTime.now().subtract(const Duration(days: 1)),
        updatedAt: DateTime.now().subtract(const Duration(days: 1)),
      ),
    ];

    // 搜索过滤
    var filteredDatasets = allDatasets;
    if (search.isNotEmpty) {
      final keyword = search.toLowerCase();
      filteredDatasets = allDatasets
          .where(
            (dataset) =>
                dataset.name.toLowerCase().contains(keyword) ||
                (dataset.description?.toLowerCase().contains(keyword) ?? false),
          )
          .toList();
    }

    const pageSize = 10;
    final start = (page - 1) * pageSize;
    final end = (start + pageSize).clamp(0, filteredDatasets.length);
    final list = filteredDatasets.sublist(start, end);

    return PaginatedDatasetResponse(
      list: list,
      total: filteredDatasets.length,
      page: page,
      pageSize: pageSize,
      totalPages: (filteredDatasets.length / pageSize).ceil(),
    );
  }

  DatasetModel _getMockDatasetDetail(int datasetId) {
    final datasets = _getMockDatasets(1, '');
    return datasets.list.firstWhere(
      (dataset) => dataset.id == datasetId,
      orElse: () => throw Exception('Dataset not found'),
    );
  }

  PaginatedImageResponse _getMockDatasetImages(
    int datasetId,
    int page,
    ImageType? imageType,
    String search,
  ) {
    final imageUrls = [
      'https://images.unsplash.com/photo-1500534314209-a25ddb2bd429?w=800&h=600&fit=crop',
      'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800&h=600&fit=crop',
      'https://images.unsplash.com/photo-1497366216548-37526070297c?w=800&h=600&fit=crop',
      'https://images.unsplash.com/photo-1519681393784-d120267933ba?w=800&h=600&fit=crop',
      'https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?w=800&h=600&fit=crop',
      'https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=800&h=600&fit=crop',
      'https://images.unsplash.com/photo-1472214103451-9374bd1c798e?w=800&h=600&fit=crop',
      'https://images.unsplash.com/photo-1426604966848-d7adac402bff?w=800&h=600&fit=crop',
      'https://images.unsplash.com/photo-1501594907352-04cda38ebc29?w=800&h=600&fit=crop',
      'https://images.unsplash.com/photo-1469474968028-56623f02e42e?w=800&h=600&fit=crop',
    ];

    final datasetDetail = _getMockDatasetDetail(datasetId);
    final allImages = <ImageModel>[];

    for (var i = 0; i < datasetDetail.totalImages; i++) {
      final type = switch (i % 3) {
        0 => ImageType.foggy,
        1 => ImageType.clear,
        _ => ImageType.annotated,
      };

      // 根据数据集的实际分布调整图片类型数量
      ImageType actualType;
      if (type == ImageType.foggy && i >= datasetDetail.foggyCount) {
        actualType = ImageType.clear;
      } else if (type == ImageType.clear &&
          (datasetDetail.foggyCount + (i - datasetDetail.foggyCount)) >=
              (datasetDetail.foggyCount + datasetDetail.clearCount)) {
        actualType = ImageType.annotated;
      } else {
        actualType = type;
      }

      allImages.add(
        ImageModel(
          id: datasetId * 1000 + i,
          datasetId: datasetId,
          filename:
              "${datasetDetail.name.replaceAll(' ', '_')}_${actualType.name}_${(i + 1).toString().padLeft(4, '0')}.jpg",
          imageUrl: imageUrls[i % imageUrls.length],
          imageType: actualType,
          width: 1920,
          height: 1080,
          fileSize: 1024000 + (i * 100000), // 1MB - 3MB
          tags: '${actualType.name},${datasetDetail.name}',
          description: '${datasetDetail.name}中的${actualType.displayName}图像',
          createdAt: DateTime.now().subtract(Duration(days: 30 - i)),
        ),
      );
    }

    // 类型过滤
    var filteredImages = allImages;
    if (imageType != null) {
      filteredImages = allImages
          .where((img) => img.imageType == imageType)
          .toList();
    }

    // 搜索过滤
    if (search.isNotEmpty) {
      final keyword = search.toLowerCase();
      filteredImages = filteredImages
          .where(
            (img) =>
                img.filename.toLowerCase().contains(keyword) ||
                (img.tags?.toLowerCase().contains(keyword) ?? false) ||
                (img.description?.toLowerCase().contains(keyword) ?? false),
          )
          .toList();
    }

    const pageSize = 20;
    final start = (page - 1) * pageSize;
    final end = (start + pageSize).clamp(0, filteredImages.length);
    final list = filteredImages.sublist(start, end);

    return PaginatedImageResponse(
      list: list,
      total: filteredImages.length,
      page: page,
      pageSize: pageSize,
      totalPages: (filteredImages.length / pageSize).ceil(),
    );
  }
}
