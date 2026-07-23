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
///
/// 业务状态码由 [ResponseInterceptor] 统一拦截：
/// code == '00000' 放行，否则 reject 抛出 ApiException，
/// 因此此处响应均为成功，直接读取 `data` 字段即可。
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

    final data = response.data!['data'] as Map<String, dynamic>?;
    final list = (data?['list'] as List<dynamic>? ?? [])
        .map((e) => DatasetModel.fromJson(e as Map<String, dynamic>))
        .toList();
    return list;
  }

  /// 获取数据集详情
  ///
  /// GET /datasets/{id}
  Future<DatasetModel> getDatasetDetail(int datasetId) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.datasets}/$datasetId',
    );

    return DatasetModel.fromJson(response.data!['data'] as Map<String, dynamic>);
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

    final data = response.data!['data'] as Map<String, dynamic>;
    final list = (data['list'] as List<dynamic>? ?? [])
        .map((e) => DatasetItemModel.fromJson(e as Map<String, dynamic>))
        .toList();
    final total = data['total'] as int? ?? 0;
    return PageResult(list: list, total: total);
  }

  /// 获取数据项详情（含图片文件）
  ///
  /// GET /dataset-items/{id}
  Future<DatasetItemModel> getItemDetail(int itemId) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.datasetItems}/$itemId',
    );

    return DatasetItemModel.fromJson(
        response.data!['data'] as Map<String, dynamic>);
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

    // 将数据项的清晰图与有雾图展开为图片列表
    final images = <ImageModel>[];
    for (final item in itemsResponse.list) {
      for (final image in item.allImages) {
        // 按类型过滤
        if (imageType != null && image.imageType != imageType) continue;

        images.add(ImageModel.fromItemImage(image, datasetId, item.createTime));
      }
    }

    return PageResult(list: images, total: itemsResponse.total);
  }
}
