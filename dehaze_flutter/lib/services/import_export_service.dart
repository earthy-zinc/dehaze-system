import 'package:dio/dio.dart';

import '../core/constants/api_constants.dart';
import '../core/network/page_result.dart';
import '../models/import_export_model.dart';

/// 导入导出服务
///
/// 对齐 JS SDK ImportExportAPI 全部方法。
class ImportExportService {
  const ImportExportService(this._dio);

  final Dio _dio;

  // ==================== 导入 ====================

  /// 导入数据（上传文件）
  ///
  /// [formData] 需包含 file 字段及模块特定参数（如 mode、async、deptId 等）。
  Future<ImportRecordVO> importData(FormData formData) async {
    final response = await _dio.post<Map<String, dynamic>>(
      '${ApiConstants.importExport}/import',
      data: formData,
      options: Options(
        headers: {'Content-Type': 'multipart/form-data'},
      ),
    );
    return ImportRecordVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 分页查询导入记录
  Future<PageResult<ImportRecordVO>> getImportRecords(
    ImportRecordQuery query,
  ) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.importExport}/import/records',
      queryParameters: query.toQuery(),
    );
    final data = response.data!['data'] as Map<String, dynamic>;
    final list = (data['list'] as List<dynamic>?)
            ?.map(
              (e) => ImportRecordVO.fromJson(e as Map<String, dynamic>),
            )
            .toList() ??
        [];
    return PageResult<ImportRecordVO>(
      list: list,
      total: (data['total'] as int?) ?? 0,
    );
  }

  /// 获取导入模板元数据
  Future<ImportTemplateVO> getImportTemplate(String type) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.importExport}/import/template/$type',
    );
    return ImportTemplateVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  // ==================== 导出 ====================

  /// 导出数据
  ///
  /// [params] 包含 format、async、fields 及模块特定筛选条件。
  Future<ExportRecordVO> exportData(Map<String, dynamic> params) async {
    final response = await _dio.post<Map<String, dynamic>>(
      '${ApiConstants.importExport}/export',
      data: params,
    );
    return ExportRecordVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 分页查询导出记录
  Future<PageResult<ExportRecordVO>> getExportRecords(
    ExportRecordQuery query,
  ) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.importExport}/export/records',
      queryParameters: query.toQuery(),
    );
    final data = response.data!['data'] as Map<String, dynamic>;
    final list = (data['list'] as List<dynamic>?)
            ?.map(
              (e) => ExportRecordVO.fromJson(e as Map<String, dynamic>),
            )
            .toList() ??
        [];
    return PageResult<ExportRecordVO>(
      list: list,
      total: (data['total'] as int?) ?? 0,
    );
  }

  /// 下载导出文件（流式响应）
  Future<Response<List<int>>> downloadExport(int id) async {
    return _dio.get<List<int>>(
      '${ApiConstants.importExport}/export/$id/download',
      options: Options(responseType: ResponseType.stream),
    );
  }
}
