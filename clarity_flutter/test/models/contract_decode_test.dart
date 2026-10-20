import 'package:clarity_flutter/models/announcement_model.dart';
import 'package:clarity_flutter/models/api_key_model.dart';
import 'package:clarity_flutter/models/feedback_model.dart';
import 'package:clarity_flutter/models/member_model.dart';
import 'package:clarity_flutter/models/order_model.dart';
import 'package:clarity_flutter/models/package_model.dart';
import 'package:clarity_flutter/models/user_model.dart';
import 'package:flutter_test/flutter_test.dart';

/// 后端 BenefitVO 完整样例（会员档案内嵌权益）
Map<String, dynamic> _benefitJson() => <String, dynamic>{
  'levelCode': 'level_1',
  'levelName': '青铜',
  'growthMin': 0,
  'growthMax': 1000,
  'monthlyDehazeQuota': 50,
  'monthlyDerainQuota': 10,
  'monthlyDesnowQuota': 10,
  'monthlyLowlightQuota': 10,
  'monthlySuperResolutionQuota': 10,
  'monthlyDenoiseQuota': 10,
  'monthlyInpaintQuota': 10,
  'monthlyEvaluateQuota': 20,
  'aiCreditsDaily': 100,
  'aiCreditsMonthly': 1000,
  'multimodalLimit': 5,
  'maxDevices': 1,
  'vipGiftCredits': 200,
  'historyRetention': 30,
  'batchLimit': 5,
  'priority': 1,
  'advancedParams': 1,
  'hdExport': 1,
  'reportExport': 0,
  'batchDownload': 0,
  'sort': 1,
  'status': 1,
};

/// 契约反序列化守卫
///
/// 样例 JSON 按后端实际返回（Java/Go VO 与 JS SDK 模型）构造，
/// 覆盖曾发生漂移的模型，防止再次出现"字段名/类型与后端不一致导致解析失败"。
void main() {
  group('当前用户（/auth/me）', () {
    test('roles/perms 映射到 roleNames/permissions 并驱动权限判定', () {
      final user = UserModel.fromJson(<String, dynamic>{
        'userId': 2,
        'username': 'admin',
        'nickname': 'admin',
        'avatar': null,
        'roles': <String>['ROOT', 'ADMIN'],
        'perms': <String>['sys:user:add', 'order:list'],
      });

      expect(user.roleNames, <String>['ROOT', 'ADMIN']);
      expect(user.permissions.length, 2);

      // ROOT 角色放行一切
      expect(user.isRoot, true);
      expect(user.hasModulePerm('sys:role'), true);
      expect(user.hasPermission('any:perm'), true);
    });

    test('非 ROOT 用户按具体权限与模块前缀判定', () {
      final user = UserModel.fromJson(<String, dynamic>{
        'userId': 9,
        'username': 'staff',
        'nickname': 'staff',
        'roles': <String>['ADMIN'],
        'perms': <String>['sys:user:add', 'order:list'],
      });

      expect(user.isRoot, false);
      expect(user.hasModulePerm('sys:user'), true);
      expect(user.hasModulePerm('sys:role'), false);
      expect(user.hasPermission('order:list'), true);
    });
  });

  group('订单域', () {
    test('OrderPageVO 按后端字段解析', () {
      final vo = OrderPageVO.fromJson(<String, dynamic>{
        'id': 1,
        'orderNo': 'D202610020001',
        'packageName': '月卡',
        'packageLevel': 'level_1',
        'packageType': 'vip',
        'creditAmount': null,
        'payableAmount': 1990,
        'paidAmount': 1990,
        'payMethod': 'balance',
        'status': 'paid',
        'createTime': '2026-10-02 10:00:00',
        'paidTime': '2026-10-02 10:01:00',
        'packageExpireTime': '2026-11-02 10:01:00',
        'userId': 2,
        'username': 'admin',
        'originalPrice': 2990,
        'discountAmount': 1000,
        'couponAmount': 0,
      });

      expect(vo.status, OrderStatus.paid);
      expect(vo.paidAmount, 1990);
      expect(vo.couponAmount, 0);
      expect(vo.username, 'admin');
    });

    test('积分卡订单 packageLevel 缺失/为 null 时可解析', () {
      final vo = OrderPageVO.fromJson(<String, dynamic>{
        'id': 85,
        'orderNo': 'DH20261001232008994724',
        'packageName': '测试套餐_credit',
        'packageType': 'credit',
        'packageLevel': null,
        'creditAmount': 1000,
        'payableAmount': 100,
        'paidAmount': 100,
        'payMethod': 'balance',
        'status': 'completed',
        'createTime': '2026-10-01 23:20:08',
        'paidTime': '2026-10-01 23:20:09',
        'userId': 5,
        'username': 'user',
        'originalPrice': 1990,
        'discountAmount': 0,
        'couponAmount': 0,
      });

      expect(vo.packageLevel, isNull);
      expect(vo.creditAmount, 1000);
      expect(vo.status, OrderStatus.completed);
    });

    test('RefundRecordVO 按后端字段解析', () {
      final vo = RefundRecordVO.fromJson(<String, dynamic>{
        'id': 5,
        'refundNo': 'R202610020001',
        'orderId': 1,
        'orderNo': 'D202610020001',
        'userId': 2,
        'username': 'admin',
        'refundAmount': 1990,
        'reasonType': 'after_sale',
        'reason': 'after_sale:不想要了',
        'usedDays': 3,
        'usedCredits': null,
        'status': 'refunding',
        'channel': null,
        'channelRefundNo': null,
        'applyTime': '2026-10-02 11:00:00',
        'auditTime': null,
        'auditorId': null,
        'auditRemark': null,
        'refundTime': null,
        'errorMessage': null,
      });

      expect(vo.status, RefundStatus.refunding);
      expect(vo.reasonType, RefundReasonType.afterSale);
      expect(vo.refundAmount, 1990);
      expect(vo.refundNo, 'R202610020001');
    });

    test('PaymentRecordVO 按后端字段解析', () {
      final vo = PaymentRecordVO.fromJson(<String, dynamic>{
        'id': 3,
        'paymentNo': 'P202610020001',
        'channel': 'wechat',
        'amount': 1990,
        'status': 2,
        'callbackTime': '2026-10-02 10:00:30',
        'createTime': '2026-10-02 10:00:00',
      });

      expect(vo.amount, 1990);
      expect(vo.paymentNo, 'P202610020001');
      expect(vo.status, 2);
    });
  });

  group('会员域', () {
    test('BenefitVO 含完整配额字段', () {
      final vo = BenefitVO.fromJson(_benefitJson());

      expect(vo.monthlyDerainQuota, 10);
      expect(vo.maxDevices, 1);
      expect(vo.vipGiftCredits, 200);
    });

    test('MemberProfileVO 含 levelSource', () {
      final vo = MemberProfileVO.fromJson(<String, dynamic>{
        'userId': 2,
        'username': 'admin',
        'nickname': '管理员',
        'avatar': null,
        'levelCode': 'level_1',
        'levelName': '青铜',
        'levelSource': 'growth',
        'growthValue': 120,
        'nextLevelGrowth': 1000,
        'progressPercent': 12,
        'expireTime': null,
        'monthlyDehazeQuota': 50,
        'monthlyDehazeUsed': 3,
        'monthlyEvaluateQuota': 20,
        'monthlyEvaluateUsed': 1,
        'benefits': _benefitJson(),
        'status': 1,
      });

      expect(vo.levelSource, 'growth');
      expect(vo.monthlyDehazeUsed, 3);
    });
  });

  group('优惠券', () {
    test('CouponVO 按后端字段解析', () {
      final vo = CouponVO.fromJson(<String, dynamic>{
        'id': 1,
        'name': '新人券',
        'type': 'full_reduction',
        'faceValue': 500,
        'threshold': 2000,
        'validType': 'fixed',
        'validStart': '2026-10-01 00:00:00',
        'validEnd': '2026-12-31 23:59:59',
        'validDays': null,
        'totalQty': 100,
        'issuedQty': 10,
        'usedQty': 2,
        'perUserLimit': 1,
        'applicableScope': <dynamic>[1, 'vip'],
        'status': 1,
        'createTime': '2026-10-01 00:00:00',
      });

      expect(vo.type, CouponType.fullReduction);
      expect(vo.faceValue, 500);
      expect(vo.issuedQty, 10);
    });

    test('UserCouponVO 状态为 1/2/3/4', () {
      UserCouponVO decode(int status) =>
          UserCouponVO.fromJson(<String, dynamic>{
            'id': 1,
            'couponId': 1,
            'couponName': '新人券',
            'type': 'discount',
            'faceValue': 500,
            'threshold': null,
            'status': status,
            'receiveTime': '2026-10-01 00:00:00',
            'expireTime': null,
            'usedTime': null,
            'usedOrderId': null,
            'applicableScope': null,
          });

      expect(decode(1).status, UserCouponStatus.unused);
      expect(decode(2).status, UserCouponStatus.used);
      expect(decode(3).status, UserCouponStatus.expired);
      expect(decode(4).status, UserCouponStatus.locked);
    });
  });

  group('反馈', () {
    test('FeedbackPageVO 枚举为字符串/数字混合口径', () {
      final vo = FeedbackPageVO.fromJson(<String, dynamic>{
        'id': 1,
        'userId': 2,
        'username': 'admin',
        'feedbackType': 'bug',
        'title': '上传失败',
        'content': '详情',
        'status': 'processing',
        'priority': 1,
        'assigneeId': 3,
        'assigneeName': '客服',
        'relatedModule': 'dataset',
        'tags': <String>['upload'],
        'createTime': '2026-10-02 10:00:00',
        'updateTime': null,
      });

      expect(vo.feedbackType, FeedbackType.bug);
      expect(vo.status, FeedbackStatus.processing);
    });

    test('FeedbackReplyVO 回复人类型为数字 1/2', () {
      final vo = FeedbackReplyVO.fromJson(<String, dynamic>{
        'id': 1,
        'feedbackId': 1,
        'replierId': 3,
        'replierName': '客服',
        'replierType': 2,
        'content': '已修复',
        'replyType': 'resolved',
        'attachments': null,
        'createTime': '2026-10-02 12:00:00',
      });

      expect(vo.replierType, ReplierType.admin);
      expect(vo.replyType, FeedbackReplyType.resolved);
    });
  });

  group('公告与密钥', () {
    test('AnnouncementVO 按后端字段解析', () {
      final vo = AnnouncementVO.fromJson(<String, dynamic>{
        'id': 1,
        'title': '维护通知',
        'content': '内容',
        'type': 'NOTICE',
        'typeLabel': '通知',
        'importance': 2,
        'importanceLabel': '重要',
        'targetScope': 'all',
        'targetScopeLabel': '全体用户',
        'targetParams': null,
        'status': 3,
        'statusLabel': '已发送',
        'sendTime': '2026-10-02 09:00:00',
        'expireTime': null,
        'sentCount': 12,
        'createTime': '2026-10-02 08:00:00',
        'updateTime': null,
        'createBy': 2,
      });

      expect(vo.importance, 2);
      expect(vo.targetScope, AnnouncementTargetScope.all);
      expect(vo.status, 3);
      expect(vo.sentCount, 12);
      expect(vo.targetParams, isNull);
    });

    test('AnnouncementForm 发送范围序列化为后端小写取值', () {
      final json = AnnouncementForm(
        title: 't',
        content: 'c',
        type: 'NOTICE',
        importance: 1,
        targetScope: AnnouncementTargetScope.all,
      ).toJson();

      expect(json['targetScope'], 'all');
    });

    test('AnnouncementSendResult 只有 sentCount', () {
      final result = AnnouncementSendResult.fromJson(<String, dynamic>{
        'sentCount': 12,
      });
      expect(result.sentCount, 12);
    });

    test('ApiKeyVO 含配额字段', () {
      final vo = ApiKeyVO.fromJson(<String, dynamic>{
        'id': 1,
        'name': '默认密钥',
        'apiKey': 'dhak_xxx',
        'keyPrefix': 'dhak_1234',
        'status': 1,
        'expiresAt': null,
        'lastUsedAt': null,
        'createTime': '2026-10-01 00:00:00',
        'dailyQuota': 100,
        'monthlyQuota': 1000,
        'rpmLimit': 60,
        'modelWhitelist': <String>['qwen3-8b'],
      });

      expect(vo.keyPrefix, 'dhak_1234');
      expect(vo.modelWhitelist, <String>['qwen3-8b']);
      expect(vo.rpmLimit, 60);
    });
  });
}
