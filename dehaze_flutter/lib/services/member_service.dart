import 'package:dio/dio.dart';

import '../core/constants/api_constants.dart';
import '../core/network/page_result.dart';
import '../models/member_model.dart';

/// 会员服务（用户端 + 管理端）
///
/// 对齐 JS SDK MemberAPI 的全部方法。
class MemberService {
  const MemberService(this._dio);

  final Dio _dio;

  // ==================== 用户端 ====================

  /// 当前用户会员信息
  Future<MemberProfileVO> getProfile() async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.membersProfile,
    );
    return MemberProfileVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 成长值变动明细
  Future<PageResult<GrowthLogVO>> getGrowthLogs(GrowthLogQuery query) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.membersGrowthLogs,
      queryParameters: query.toJson(),
    );
    final data = response.data!['data'] as Map<String, dynamic>;
    final list = (data['list'] as List<dynamic>)
        .map((e) => GrowthLogVO.fromJson(e as Map<String, dynamic>))
        .toList();
    return PageResult(list: list, total: data['total'] as int);
  }

  /// 每日签到
  Future<SignInResultVO> signIn() async {
    final response = await _dio.post<Map<String, dynamic>>(
      ApiConstants.membersSignIn,
    );
    return SignInResultVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 签到日历
  Future<SignInCalendarVO> getSignInCalendar({
    required int year,
    required int month,
  }) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.membersSignInCalendar,
      queryParameters: {'year': year, 'month': month},
    );
    return SignInCalendarVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  // ==================== 管理端 ====================

  /// 会员分页列表
  Future<PageResult<MemberPageVO>> getPage(MemberQuery query) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.membersPage,
      queryParameters: query.toJson(),
    );
    final data = response.data!['data'] as Map<String, dynamic>;
    final list = (data['list'] as List<dynamic>)
        .map((e) => MemberPageVO.fromJson(e as Map<String, dynamic>))
        .toList();
    return PageResult(list: list, total: data['total'] as int);
  }

  /// 会员详情
  Future<MemberDetailVO> getDetail(int userId) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.members}/$userId',
    );
    return MemberDetailVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 等级调整
  Future<void> adjustLevel(int userId, MemberLevelAdjustForm form) async {
    await _dio.put<Map<String, dynamic>>(
      '${ApiConstants.members}/$userId/level',
      data: form.toJson(),
    );
  }

  /// 成长值调整
  Future<void> adjustGrowth(int userId, MemberGrowthAdjustForm form) async {
    await _dio.put<Map<String, dynamic>>(
      '${ApiConstants.members}/$userId/growth',
      data: form.toJson(),
    );
  }

  /// 冻结/解冻
  Future<void> updateStatus(int userId, MemberStatusForm form) async {
    await _dio.put<Map<String, dynamic>>(
      '${ApiConstants.members}/$userId/status',
      data: form.toJson(),
    );
  }

  /// 权益配置列表
  Future<List<BenefitVO>> getBenefits() async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.membersBenefits,
    );
    final list = response.data!['data'] as List<dynamic>;
    return list
        .map((e) => BenefitVO.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  /// 修改权益配置
  Future<void> updateBenefit(String levelCode, BenefitForm form) async {
    await _dio.put<Map<String, dynamic>>(
      '${ApiConstants.membersBenefits}/$levelCode',
      data: form.toJson(),
    );
  }
}
