import 'package:dio/dio.dart';

import '../core/constants/api_constants.dart';
import '../core/network/page_result.dart';
import '../core/types/option_type.dart';
import '../models/dataset_model.dart';

// ==================== 数据集服务 ====================

class DatasetService {
  const DatasetService(this._dio);

  final Dio _dio;

  /// 分页查询数据集列表
  Future<PageResult<Dataset>> getList([DatasetQuery? queryParams]) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.datasets,
      queryParameters: queryParams?.toJson(),
    );
    final data = response.data!['data'] as Map<String, dynamic>;
    return PageResult<Dataset>(
      list: (data['list'] as List<dynamic>)
          .map((e) => Dataset.fromJson(e as Map<String, dynamic>))
          .toList(),
      total: data['total'] as int,
    );
  }

  /// 获取子数据集列表（懒加载）
  Future<List<Dataset>> getChildren(int parentId) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.datasetsChildren}/$parentId',
    );
    final data = response.data!['data'] as List<dynamic>;
    return data
        .map((e) => Dataset.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  /// 获取数据集下拉选项列表
  Future<List<OptionType>> getOptions() async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.datasetsOptions,
    );
    final data = response.data!['data'] as List<dynamic>;
    return data
        .map((e) => OptionType.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  /// 根据ID获取数据集详细信息
  Future<Dataset> getDatasetInfoById(int id) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.datasets}/$id',
    );
    return Dataset.fromJson(response.data!['data'] as Map<String, dynamic>);
  }

  /// 新增数据集
  Future<int> add(DatasetAddForm form) async {
    final response = await _dio.post<Map<String, dynamic>>(
      ApiConstants.datasets,
      data: form.toJson(),
    );
    return response.data!['data'] as int;
  }

  /// 修改数据集信息
  Future<Dataset> update(int id, DatasetUpdateForm form) async {
    final response = await _dio.put<Map<String, dynamic>>(
      '${ApiConstants.datasets}/$id',
      data: form.toJson(),
    );
    return Dataset.fromJson(response.data!['data'] as Map<String, dynamic>);
  }

  /// 删除单个数据集
  Future<void> deleteById(int id) async {
    await _dio.delete<Map<String, dynamic>>(
      '${ApiConstants.datasets}/$id',
    );
  }

  /// 批量删除数据集
  Future<BatchDeleteResultVO> batchDelete(BatchDeleteForm form) async {
    final response = await _dio.delete<Map<String, dynamic>>(
      ApiConstants.datasetsBatch,
      data: form.toJson(),
    );
    return BatchDeleteResultVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }
}

// ==================== 数据项服务 ====================

class DatasetItemService {
  const DatasetItemService(this._dio);

  final Dio _dio;

  /// 分页查询数据项列表
  Future<PageResult<DatasetItemVO>> getList([DatasetItemQuery? queryParams]) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.datasetItems,
      queryParameters: queryParams?.toJson(),
    );
    final data = response.data!['data'] as Map<String, dynamic>;
    return PageResult<DatasetItemVO>(
      list: (data['list'] as List<dynamic>)
          .map((e) => DatasetItemVO.fromJson(e as Map<String, dynamic>))
          .toList(),
      total: data['total'] as int,
    );
  }

  /// 创建空数据项
  Future<DatasetItemVO> add(DatasetItemCreateForm form) async {
    final response = await _dio.post<Map<String, dynamic>>(
      ApiConstants.datasetItems,
      data: form.toJson(),
    );
    return DatasetItemVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 获取数据项详情
  Future<DatasetItemVO> getById(int id) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.datasetItems}/$id',
    );
    return DatasetItemVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 修改数据项信息
  Future<DatasetItemVO> update(int id, DatasetItemUpdateForm form) async {
    final response = await _dio.put<Map<String, dynamic>>(
      '${ApiConstants.datasetItems}/$id',
      data: form.toJson(),
    );
    return DatasetItemVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 删除数据项
  Future<void> deleteById(int id) async {
    await _dio.delete<Map<String, dynamic>>(
      '${ApiConstants.datasetItems}/$id',
    );
  }

  /// 创建数据项并上传配对图片
  Future<DatasetItemVO> uploadImagePair(FormData formData) async {
    final response = await _dio.post<Map<String, dynamic>>(
      ApiConstants.datasetItemsUpload,
      data: formData,
      options: Options(
        headers: {'Content-Type': 'multipart/form-data'},
      ),
    );
    return DatasetItemVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 批量创建数据项并上传图片
  Future<BatchUploadResultVO> batchUpload(FormData formData) async {
    final response = await _dio.post<Map<String, dynamic>>(
      ApiConstants.datasetItemsBatch,
      data: formData,
      options: Options(
        headers: {'Content-Type': 'multipart/form-data'},
      ),
    );
    return BatchUploadResultVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 批量删除数据项
  Future<BatchOperationResultVO> batchDelete(BatchDeleteForm form) async {
    final response = await _dio.delete<Map<String, dynamic>>(
      ApiConstants.datasetItemsBatch,
      data: form.toJson(),
    );
    return BatchOperationResultVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }
}

// ==================== 图片文件服务 ====================

class ItemFileService {
  const ItemFileService(this._dio);

  final Dio _dio;

  /// 上传数据项图片
  Future<ImageUrlVO> upload(FormData formData) async {
    final response = await _dio.post<Map<String, dynamic>>(
      ApiConstants.itemFiles,
      data: formData,
      options: Options(
        headers: {'Content-Type': 'multipart/form-data'},
      ),
    );
    return ImageUrlVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 获取图片详细信息
  Future<ImageUrlVO> getById(int id) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.itemFiles}/$id',
    );
    return ImageUrlVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 修改图片信息
  Future<void> update(int id, ItemFileUpdateForm form) async {
    await _dio.put<Map<String, dynamic>>(
      '${ApiConstants.itemFiles}/$id',
      data: form.toJson(),
    );
  }

  /// 删除图片
  Future<void> deleteById(int id) async {
    await _dio.delete<Map<String, dynamic>>(
      '${ApiConstants.itemFiles}/$id',
    );
  }

  /// 批量删除图片
  Future<BatchDeleteResultVO> batchDelete(BatchDeleteForm form) async {
    final response = await _dio.delete<Map<String, dynamic>>(
      ApiConstants.itemFilesBatch,
      data: form.toJson(),
    );
    return BatchDeleteResultVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }
}
