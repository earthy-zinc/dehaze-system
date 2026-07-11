import 'package:dio/dio.dart';

import '../../../core/constants/api_constants.dart';
import '../../../core/network/page_result.dart';
import '../models/dataset_model.dart';

/// 数据集服务
///
/// 封装数据集管理相关 API：
/// - getDatasets: 获取数据集列表（树形）
/// - getDatasetDetail: 获取数据集详情
/// - getDatasetItems: 分页查询数据项
/// - getItemFiles: 获取数据项图片
class DatasetService {
  const DatasetService(this._dio);

  final Dio _dio;

  /// 获取数据集列表（树形）
  ///
  /// GET /datasets
  Future<List<DatasetModel>> getDatasets({String? keywords}) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.datasets,
      queryParameters: {
        if (keywords != null && keywords.isNotEmpty) 'keywords': keywords,
      },
    );

    final result = response.data!;
    if (result['code']?.toString() == ApiConstants.successCode) {
      final list = result['data'] as List<dynamic>? ?? [];
      return list
          .map((e) => DatasetModel.fromJson(e as Map<String, dynamic>))
          .toList();
    }
    throw Exception(result['msg'] ?? '获取数据集列表失败');
  }

  /// 获取数据集详情
  ///
  /// GET /datasets/{id}
  Future<DatasetModel> getDatasetDetail(int datasetId) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.datasets}/$datasetId',
    );

    final result = response.data!;
    if (result['code']?.toString() == ApiConstants.successCode) {
      return DatasetModel.fromJson(result['data'] as Map<String, dynamic>);
    }
    throw Exception(result['msg'] ?? '获取数据集详情失败');
  }

  /// 获取数据集下拉选项
  ///
  /// GET /datasets/options
  Future<List<DatasetModel>> getDatasetOptions() async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.datasetsOptions,
    );

    final result = response.data!;
    if (result['code']?.toString() == ApiConstants.successCode) {
      final list = result['data'] as List<dynamic>? ?? [];
      return list
          .map((e) => DatasetModel.fromJson(e as Map<String, dynamic>))
          .toList();
    }
    throw Exception(result['msg'] ?? '获取数据集选项失败');
  }

  /// 分页查询数据项列表
  ///
  /// GET /dataset-items?datasetId={id}&pageNum={n}&pageSize={s}
  Future<PageResult<DatasetItemModel>> getDatasetItems({
    required int datasetId,
    int pageNum = 1,
    int pageSize = 20,
    String? keywords,
  }) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.datasetItems,
      queryParameters: {
        'datasetId': datasetId,
        'pageNum': pageNum,
        'pageSize': pageSize,
        if (keywords != null && keywords.isNotEmpty) 'keywords': keywords,
      },
    );

    final result = response.data!;
    if (result['code']?.toString() == ApiConstants.successCode) {
      final data = result['data'] as Map<String, dynamic>;
      final list = (data['list'] as List<dynamic>? ?? [])
          .map((e) => DatasetItemModel.fromJson(e as Map<String, dynamic>))
          .toList();
      final total = data['total'] as int? ?? 0;
      return PageResult(list: list, total: total);
    }
    throw Exception(result['msg'] ?? '获取数据项列表失败');
  }

  /// 获取数据项详情（含图片文件）
  ///
  /// GET /dataset-items/{id}
  Future<DatasetItemModel> getItemDetail(int itemId) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.datasetItems}/$itemId',
    );

    final result = response.data!;
    if (result['code']?.toString() == ApiConstants.successCode) {
      return DatasetItemModel.fromJson(result['data'] as Map<String, dynamic>);
    }
    throw Exception(result['msg'] ?? '获取数据项详情失败');
  }

  /// 获取数据集下的所有图片（通过数据项）
  ///
  /// 分页获取数据项，然后提取图片文件
  Future<PageResult<ImageModel>> getDatasetImages({
    required int datasetId,
    int pageNum = 1,
    int pageSize = 20,
    ImageType? imageType,
    String? keywords,
  }) async {
    final itemsResponse = await getDatasetItems(
      datasetId: datasetId,
      pageNum: pageNum,
      pageSize: pageSize,
      keywords: keywords,
    );

    // 将数据项的文件展开为图片列表
    final images = <ImageModel>[];
    for (final item in itemsResponse.list) {
      for (final file in item.files) {
        // 按类型过滤
        if (imageType != null && file.imageType != imageType) continue;

        images.add(ImageModel(
          id: file.id,
          datasetId: datasetId,
          filename: file.fileName ?? 'image_${file.id}',
          imageUrl: file.fileUrl,
          imageType: file.imageType,
          width: file.width,
          height: file.height,
          fileSize: file.fileSize,
          createdAt: item.createTime,
        ));
      }
    }

    return PageResult(list: images, total: itemsResponse.total);
  }
}
