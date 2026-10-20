import 'package:dio/dio.dart';

import '../core/constants/api_constants.dart';
import '../models/dept_model.dart';

class DeptService {
  const DeptService(this._dio);
  final Dio _dio;

  /// 获取部门树形列表
  Future<List<Dept>> getList({String? name, int? status}) async {
    final query = DeptQuery(name: name, status: status);
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.depts,
      queryParameters: query.toQuery(),
    );
    final list = response.data!['data'] as List<dynamic>? ?? [];
    return list.map((e) => Dept.fromJson(e as Map<String, dynamic>)).toList();
  }

  /// 获取部门下拉选项
  Future<List<DeptOption>> getOptions() async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.deptsOptions,
    );
    final list = response.data!['data'] as List<dynamic>? ?? [];
    return list
        .map((e) => DeptOption.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  /// 获取部门详情
  Future<Dept> getById(int id) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.depts}/$id',
    );
    return Dept.fromJson(response.data!['data'] as Map<String, dynamic>);
  }

  /// 添加部门
  Future<int> add(DeptForm form) async {
    final response = await _dio.post<Map<String, dynamic>>(
      ApiConstants.depts,
      data: form.toJson(),
    );
    return response.data!['data'] as int;
  }

  /// 修改部门
  Future<void> update(int id, DeptForm form) async {
    await _dio.put<Map<String, dynamic>>(
      '${ApiConstants.depts}/$id',
      data: form.toJson(),
    );
  }

  /// 删除部门
  Future<void> delete(int id) async {
    await _dio.delete<Map<String, dynamic>>('${ApiConstants.depts}/$id');
  }
}
