import 'package:dio/dio.dart';

import '../core/constants/api_constants.dart';
import '../core/types/option_type.dart';
import '../models/algorithm_model.dart';
import '../models/prediction_model.dart';

/// 算法服务
///
/// 封装算法管理相关 API，对齐 JS SDK AlgorithmAPI 全部方法：
/// - 管理端：CRUD、审核、版本管理、监控、删除
/// - 用户端 select/*：对比、树、详情、测试、搜索
class AlgorithmService {
  const AlgorithmService(this._dio);

  final Dio _dio;

  // ==========================================================================
  // 管理端 API
  // ==========================================================================

  /// 获取算法树形列表（管理端，GET /algorithms）
  Future<List<AlgorithmModel>> getList({String? keywords}) async {
    final query = AlgorithmQuery(keywords: keywords);
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.algorithm,
      queryParameters: query.toQuery(),
    );
    final list = response.data!['data'] as List<dynamic>? ?? [];
    return list
        .map((e) => AlgorithmModel.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  /// 获取模型下拉选项列表（GET /algorithms/options）
  Future<List<OptionType>> getOption() async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.algorithmOptions,
    );
    final list = response.data!['data'] as List<dynamic>? ?? [];
    return list
        .map((e) => OptionType.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  /// 获取所有算法扁平列表（不分页，不构建树形，GET /algorithms/list）
  Future<List<AlgorithmModel>> listAll() async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.algorithmList,
    );
    final list = response.data!['data'] as List<dynamic>? ?? [];
    return list
        .map((e) => AlgorithmModel.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  /// 获取算法详情（GET /algorithms/{id}）
  Future<AlgorithmModel> getAlgorithmInfoById(int id) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.algorithm}/$id',
    );
    return AlgorithmModel.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 新增算法（POST /algorithms）
  Future<void> add(Map<String, dynamic> data) async {
    await _dio.post<Map<String, dynamic>>(
      ApiConstants.algorithm,
      data: data,
    );
  }

  /// 修改算法（PUT /algorithms/{id}）
  Future<void> update(int id, Map<String, dynamic> data) async {
    await _dio.put<Map<String, dynamic>>(
      '${ApiConstants.algorithm}/$id',
      data: data,
    );
  }

  /// 修改算法状态（PUT /algorithms/{id}/status）
  Future<void> updateStatus(int id, int status) async {
    await _dio.put<Map<String, dynamic>>(
      '${ApiConstants.algorithm}/$id/status',
      data: {'status': status},
    );
  }

  /// 审核算法（通过/驳回，PUT /algorithms/{id}/audit）
  Future<void> auditAlgorithm(int id, AlgorithmAuditForm data) async {
    await _dio.put<Map<String, dynamic>>(
      '${ApiConstants.algorithm}/$id/audit',
      data: data.toJson(),
    );
  }

  /// 获取算法版本历史（GET /algorithms/{id}/versions）
  Future<List<AlgorithmVersionVO>> getVersions(int id) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.algorithm}/$id/versions',
    );
    final list = response.data!['data'] as List<dynamic>? ?? [];
    return list
        .map((e) => AlgorithmVersionVO.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  /// 新增算法版本（POST /algorithms/{id}/version）
  Future<void> addVersion(int id, AlgorithmVersionForm data) async {
    await _dio.post<Map<String, dynamic>>(
      '${ApiConstants.algorithm}/$id/version',
      data: data.toJson(),
    );
  }

  /// 版本回滚（POST /algorithms/{id}/rollback?versionId=...）
  Future<void> rollbackVersion(int id, int versionId) async {
    await _dio.post<Map<String, dynamic>>(
      '${ApiConstants.algorithm}/$id/rollback',
      queryParameters: {'versionId': versionId},
    );
  }

  /// 获取算法监控数据（GET /algorithms/{id}/monitor）
  Future<AlgorithmMonitorVO> getMonitorData(int id) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.algorithm}/$id/monitor',
    );
    return AlgorithmMonitorVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 获取算法统计报表（GET /algorithms/{id}/monitor/stats）
  Future<List<AlgorithmMonitorStatsItemVO>> getMonitorStats(int id) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.algorithm}/$id/monitor/stats',
    );
    final list = response.data!['data'] as List<dynamic>? ?? [];
    return list
        .map((e) =>
            AlgorithmMonitorStatsItemVO.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  /// 删除算法（DELETE /algorithms?ids=1,2,3）
  Future<void> deleteByIds(List<int> ids) async {
    await _dio.delete<Map<String, dynamic>>(
      ApiConstants.algorithm,
      queryParameters: {'ids': ids.join(',')},
    );
  }

  // ==========================================================================
  // 用户端 select/* API
  // ==========================================================================

  /// 算法对比（POST /algorithms/select/compare）
  Future<List<AlgorithmCompareVO>> compare(AlgorithmCompareForm data) async {
    final response = await _dio.post<Map<String, dynamic>>(
      ApiConstants.algorithmSelectCompare,
      data: data.toJson(),
    );
    final list = response.data!['data'] as List<dynamic>? ?? [];
    return list
        .map((e) => AlgorithmCompareVO.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  /// 获取算法选择树（仅已发布算法，GET /algorithms/select/tree）
  Future<List<AlgorithmSelectNodeVO>> tree() async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.algorithmSelectTree,
    );
    final list = response.data!['data'] as List<dynamic>? ?? [];
    return list
        .map((e) =>
            AlgorithmSelectNodeVO.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  /// 获取算法详情（含样例效果图、评分、使用次数，GET /algorithms/select/{id}）
  Future<AlgorithmDetailVO> getSelectDetail(int id) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.algorithmSelect}/$id',
    );
    return AlgorithmDetailVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 上传自定义图片测试算法效果（POST /algorithms/select/{id}/test）
  Future<PredictionResponse> test(int id, AlgorithmTestForm data) async {
    final response = await _dio.post<Map<String, dynamic>>(
      '${ApiConstants.algorithmSelect}/$id/test',
      data: data.toJson(),
    );
    return PredictionResponse.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 搜索算法（关键词/拼音/标签，GET /algorithms/select/search?keyword=...）
  Future<List<AlgorithmSelectNodeVO>> search(String keyword) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.algorithmSelectSearch,
      queryParameters: {'keyword': keyword},
    );
    final list = response.data!['data'] as List<dynamic>? ?? [];
    return list
        .map((e) =>
            AlgorithmSelectNodeVO.fromJson(e as Map<String, dynamic>))
        .toList();
  }
}
