import 'package:dio/dio.dart';

import '../core/constants/api_constants.dart';
import '../core/network/page_result.dart';
import '../models/user_model.dart';

/// 用户服务（强类型）
///
/// 管理端用户 CRUD，对齐 JS SDK UserAPI。
class UserService {
  const UserService(this._dio);
  final Dio _dio;

  /// 管理端 - 用户分页
  Future<PageResult<UserPageVO>> getPage(UserQuery query) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.usersPage,
      queryParameters: query.toQuery(),
    );
    final data = response.data!['data'] as Map<String, dynamic>;
    final list = (data['list'] as List<dynamic>)
        .map((e) => UserPageVO.fromJson(e as Map<String, dynamic>))
        .toList();
    return PageResult(list: list, total: data['total'] as int);
  }

  /// 管理端 - 用户详情
  Future<UserDetail> getById(int id) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.users}/$id',
    );
    return UserDetail.fromJson(response.data!['data'] as Map<String, dynamic>);
  }

  /// 管理端 - 新增用户
  Future<int> add(UserForm form) async {
    final response = await _dio.post<Map<String, dynamic>>(
      ApiConstants.users,
      data: form.toJson(),
    );
    return response.data!['data'] as int;
  }

  /// 管理端 - 更新用户
  Future<void> update(int id, UserForm form) async {
    await _dio.put<Map<String, dynamic>>(
      '${ApiConstants.users}/$id',
      data: form.toJson(),
    );
  }

  /// 管理端 - 重置密码
  Future<void> updatePassword(int id, String password) async {
    await _dio.patch<Map<String, dynamic>>(
      '${ApiConstants.users}/$id/password',
      data: {'password': password},
    );
  }

  /// 管理端 - 修改状态
  Future<void> updateStatus(int id, int status) async {
    await _dio.patch<Map<String, dynamic>>(
      '${ApiConstants.users}/$id/status',
      queryParameters: {'status': status},
    );
  }

  /// 管理端 - 批量删除
  Future<void> deleteByIds(List<int> ids) async {
    await _dio.delete<Map<String, dynamic>>(
      ApiConstants.users,
      queryParameters: {'ids': ids.join(',')},
    );
  }
}
