import 'package:dio/dio.dart';

import '../core/constants/api_constants.dart';
import '../core/network/page_result.dart';
import '../models/role_model.dart';

/// 角色服务（强类型）
///
/// 对齐 JS SDK RoleAPI。
class RoleService {
  const RoleService(this._dio);
  final Dio _dio;

  /// 获取角色列表
  Future<List<Role>> getList() async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.roles,
    );
    final list = response.data!['data'] as List<dynamic>? ?? [];
    return list.map((e) => Role.fromJson(e as Map<String, dynamic>)).toList();
  }

  /// 获取角色分页数据
  Future<PageResult<RolePageVO>> getPage(RoleQuery query) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.rolesPage,
      queryParameters: query.toQuery(),
    );
    final data = response.data!['data'] as Map<String, dynamic>;
    final list = (data['list'] as List<dynamic>?)
            ?.map((e) => RolePageVO.fromJson(e as Map<String, dynamic>))
            .toList() ??
        [];
    return PageResult(
      list: list,
      total: (data['total'] as num?)?.toInt() ?? 0,
    );
  }

  /// 获取角色下拉选项
  Future<List<RoleOption>> getOptions() async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.rolesOptions,
    );
    final list = response.data!['data'] as List<dynamic>? ?? [];
    return list
        .map((e) => RoleOption.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  /// 根据 ID 获取角色详情
  Future<Role> getById(int id) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.roles}/$id',
    );
    return Role.fromJson(response.data!['data'] as Map<String, dynamic>);
  }

  /// 新增角色
  Future<int> add(RoleForm form) async {
    final response = await _dio.post<Map<String, dynamic>>(
      ApiConstants.roles,
      data: form.toJson(),
    );
    return (response.data!['data'] as num).toInt();
  }

  /// 更新角色
  Future<void> update(int id, RoleForm form) async {
    await _dio.put<Map<String, dynamic>>(
      '${ApiConstants.roles}/$id',
      data: form.toJson(),
    );
  }

  /// 批量删除角色
  Future<void> delete(List<int> ids) async {
    await _dio.delete<Map<String, dynamic>>(
      ApiConstants.roles,
      queryParameters: {'ids': ids.join(',')},
    );
  }

  /// 获取角色的菜单ID列表
  Future<List<int>> getRoleMenuIds(int roleId) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.roles}/$roleId/menuIds',
    );
    final list = response.data!['data'] as List<dynamic>? ?? [];
    return list.map((e) => e as int).toList();
  }

  /// 更新角色菜单权限
  Future<void> updateRoleMenus(int roleId, List<int> menuIds) async {
    await _dio.patch<Map<String, dynamic>>(
      '${ApiConstants.roles}/$roleId/menus',
      data: menuIds,
    );
  }

  /// 修改角色状态
  Future<void> updateStatus(int roleId, int status) async {
    await _dio.patch<Map<String, dynamic>>(
      '${ApiConstants.roles}/$roleId/status',
      queryParameters: {'status': status},
    );
  }

  /// 批量删除角色（别名，与 delete 一致）
  Future<void> deleteByIds(List<int> ids) => delete(ids);
}
