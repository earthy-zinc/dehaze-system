import 'package:dio/dio.dart';

import '../core/constants/api_constants.dart';
import '../models/menu_model.dart';

class MenuService {
  const MenuService(this._dio);
  final Dio _dio;

  /// 获取菜单树形列表
  Future<List<Menu>> getList({String? name, int? status}) async {
    final query = MenuQuery(name: name, status: status);
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.menus,
      queryParameters: query.toQuery(),
    );
    final list = response.data!['data'] as List<dynamic>? ?? [];
    return list.map((e) => Menu.fromJson(e as Map<String, dynamic>)).toList();
  }

  /// 获取路由列表
  Future<List<RouteVO>> getRoutes() async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.menusRoutes,
    );
    final list = response.data!['data'] as List<dynamic>? ?? [];
    return list
        .map((e) => RouteVO.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  /// 获取菜单下拉选项
  Future<List<MenuOption>> getOptions() async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.menusOptions,
    );
    final list = response.data!['data'] as List<dynamic>? ?? [];
    return list
        .map((e) => MenuOption.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  /// 获取菜单详情
  Future<Menu> getById(int id) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.menus}/$id',
    );
    return Menu.fromJson(response.data!['data'] as Map<String, dynamic>);
  }

  /// 添加菜单
  Future<int> add(MenuForm form) async {
    final response = await _dio.post<Map<String, dynamic>>(
      ApiConstants.menus,
      data: form.toJson(),
    );
    return response.data!['data'] as int;
  }

  /// 修改菜单
  Future<void> update(int id, MenuForm form) async {
    await _dio.put<Map<String, dynamic>>(
      '${ApiConstants.menus}/$id',
      data: form.toJson(),
    );
  }

  /// 删除菜单
  Future<void> delete(int id) async {
    await _dio.delete<Map<String, dynamic>>('${ApiConstants.menus}/$id');
  }

  /// 批量删除菜单
  Future<void> deleteByIds(List<int> ids) async {
    await _dio.delete<Map<String, dynamic>>(
      ApiConstants.menus,
      queryParameters: {'ids': ids.join(',')},
    );
  }
}
