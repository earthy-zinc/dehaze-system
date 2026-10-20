import 'package:clarity_flutter/core/auth/permission_match.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('matchModulePerm', () {
    test('命中同模块任一具体权限', () {
      expect(matchModulePerm(['sys:user:query', 'order:list'], 'sys:user'), true);
      expect(matchModulePerm(['sys:user:query', 'order:list'], 'order'), true);
    });

    test('未命中其他模块', () {
      expect(matchModulePerm(['sys:user:query'], 'sys:role'), false);
      expect(matchModulePerm([], 'sys:user'), false);
    });

    test('带结尾冒号同样可用', () {
      expect(matchModulePerm(['order:list'], 'order:'), true);
    });

    test('不匹配同名前缀的相邻模块（sys:role 不受 sys:roles 影响）', () {
      expect(matchModulePerm(['sys:roles:add'], 'sys:role'), false);
    });

    test('精确等于模块名也视为命中', () {
      expect(matchModulePerm(['order'], 'order'), true);
    });
  });
}
