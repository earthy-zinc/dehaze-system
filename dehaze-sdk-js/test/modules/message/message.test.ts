import { expect } from "vitest";
import {
  AnnouncementAPI,
  MessageAPI,
  MessageTemplateAPI,
  MessageSendRequest,
  NotificationSettingAPI,
  MessageVO,
} from "../../../index";
import { expectBizError } from "#/utils/assertion";
import { login } from "#/utils/auth";
import { createAnnouncementForm, createMessageSendRequest } from "#/factories/message";
import { USERS } from "#/factories/constants";

describe("消息通知模块接口测试", () => {
  let testMessageId: number;
  let testAnnouncementId: number;
  const createdMessageIds: number[] = [];

  // 发送单条测试消息，返回消息体与首条 id（调用方需自行清理）
  async function sendMessage(overrides: Partial<MessageSendRequest> = {}) {
    const form = createMessageSendRequest(overrides);
    const { messageIds } = await MessageAPI.send(form);
    return { form, messageIds, id: messageIds[0]! };
  }

  // 按 id 列表批量删除测试消息
  async function deleteMessages(ids: number[]) {
    await MessageAPI.deleteByIds(ids.join(","));
  }

  afterAll(async () => {
    // 清理残留的测试消息
    for (const id of createdMessageIds.reverse()) {
      try {
        await MessageAPI.deleteByIds(String(id));
      } catch (e) {
        console.warn(`清理测试消息失败 id=${id}:`, e);
      }
    }
    // 清理残留的测试公告
    if (testAnnouncementId) {
      try {
        await AnnouncementAPI.deleteById(testAnnouncementId);
      } catch (e) {
        console.warn(`清理测试公告失败 id=${testAnnouncementId}:`, e);
      }
    }
  });

  describe("POST /api/v1/messages/send - 内部消息发送", () => {
    test("正向测试：不使用模板发送消息", async () => {
      const { messageIds } = await sendMessage();
      expect(messageIds.length).toBe(1);
      testMessageId = messageIds[0]!;
      createdMessageIds.push(testMessageId);
    });

    test("正向测试：批量发送给多个接收人", async () => {
      const { messageIds } = await sendMessage({ recipientIds: [1, 2] });
      expect(messageIds.length).toBe(2);
      await deleteMessages(messageIds);
    });

    test("正向测试：幂等去重，相同bizModule+bizId不重复生成", async () => {
      const bizId = "idempotent_test_" + Date.now();
      const form = createMessageSendRequest({ bizModule: "test", bizId });
      const first = await MessageAPI.send(form);
      const second = await MessageAPI.send(form);
      expect(second.messageIds).toEqual(first.messageIds);
      await deleteMessages(first.messageIds);
    });

    test("参数校验：缺少type应抛出业务错误", async () => {
      await expectBizError(
        MessageAPI.send({ recipientIds: [1], title: "x", content: "y" } as any),
        ["A0400", "A0410", "ERR_BAD_REQUEST"]
      );
    });

    test("参数校验：空接收人列表应抛出业务错误", async () => {
      await expectBizError(MessageAPI.send(createMessageSendRequest({ recipientIds: [] })), [
        "A0400",
        "A0410",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("验证：发送含 jumpUrl 的消息并验证详情", async () => {
      const jumpUrl = "/test/jump/" + Date.now();
      const { messageIds } = await sendMessage({ jumpUrl });

      const detail = await MessageAPI.getDetail(messageIds[0]!);
      expect(detail.jumpUrl).toBe(jumpUrl);

      await deleteMessages(messageIds);
    });
  });

  describe("GET /api/v1/messages - 消息列表", () => {
    test("正向测试：分页查询消息列表", async () => {
      const result = await MessageAPI.getPage({ pageNum: 1, pageSize: 20 });
      expect(Array.isArray(result.list)).toBe(true);
      expect(result.total).toBeGreaterThanOrEqual(0);
    });

    test("正向测试：按类型筛选消息", async () => {
      const result = await MessageAPI.getPage({ type: "business", pageNum: 1, pageSize: 10 });
      for (const msg of result.list) {
        expect(msg.type).toBe("business");
      }
    });

    test("正向测试：按已读状态筛选未读消息", async () => {
      const result = await MessageAPI.getPage({ readStatus: 0, pageNum: 1, pageSize: 10 });
      for (const msg of result.list) {
        expect(msg.readStatus).toBe(0);
      }
    });

    test("边界：空数据查询返回空列表", async () => {
      const result = await MessageAPI.getPage({
        type: "nonexistent_type" as any,
        pageNum: 1,
        pageSize: 10,
      });
      expect(Array.isArray(result.list)).toBe(true);
      expect(result.list.length).toBe(0);
      expect(result.total).toBe(0);
    });

    test("验证：消息按创建时间倒序排列", async () => {
      const result = await MessageAPI.getPage({ pageNum: 1, pageSize: 20 });
      if (result.list.length < 2) return;
      for (let i = 1; i < result.list.length; i++) {
        const prev = result.list[i - 1]!.createTime;
        const curr = result.list[i]!.createTime;
        if (prev && curr) {
          expect(prev >= curr).toBe(true);
        }
      }
    });
  });

  describe("GET /api/v1/messages/unread-count - 未读消息数", () => {
    test("正向测试：查询未读消息数", async () => {
      const result = await MessageAPI.getUnreadCount();
      expect(typeof result.count).toBe("number");
      expect(result.count).toBeGreaterThanOrEqual(0);
    });
  });

  describe("GET /api/v1/messages/{id} - 消息详情", () => {
    test("正向测试：查看消息详情不自动标记已读", async () => {
      const { form, messageIds, id } = await sendMessage();

      const detail = await MessageAPI.getDetail(id);
      expect(detail.id).toBe(id);
      expect(detail.title).toBe(form.title);
      expect(detail.content).toBe(form.content);
      expect(detail.readStatus).toBe(0);

      await MessageAPI.markRead(id);
      const readDetail = await MessageAPI.getDetail(id);
      expect(readDetail.readStatus).toBe(1);
      expect(readDetail.readTime).toBeDefined();

      await deleteMessages(messageIds);
    });

    test("异常：查看不存在的消息", async () => {
      await expectBizError(MessageAPI.getDetail(999999999), [
        "A0550",
        "A0401",
        "A0400",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("边界：查看他人消息应返回不存在（不暴露存在性）", async () => {
      // admin 发送给 user 的消息
      const { messageIds, id } = await sendMessage({ recipientIds: [USERS.USER.id] });

      try {
        // admin 尝试查看发给 user 的消息（admin 不是接收人）
        await expectBizError(MessageAPI.getDetail(id), [
          "A0550",
          "A0401",
          "A0400",
          "ERR_BAD_REQUEST",
        ]);
      } finally {
        await deleteMessages(messageIds);
      }
    });
  });

  describe("PATCH /api/v1/messages/{id}/_read - 标记单条已读", () => {
    test("正向测试：标记未读消息为已读", async () => {
      const { messageIds, id } = await sendMessage();

      await MessageAPI.markRead(id);
      const detail = await MessageAPI.getDetail(id);
      expect(detail.readStatus).toBe(1);

      await deleteMessages(messageIds);
    });

    test("边界：重复标记已读幂等返回成功", async () => {
      const { messageIds, id } = await sendMessage();

      await MessageAPI.markRead(id);
      await MessageAPI.markRead(id);
      const detail = await MessageAPI.getDetail(id);
      expect(detail.readStatus).toBe(1);

      await deleteMessages(messageIds);
    });
  });

  describe("PATCH /api/v1/messages/_read-all - 全部标记已读", () => {
    test("正向测试：全部标记已读并返回受影响条数", async () => {
      const form1 = createMessageSendRequest();
      const form2 = createMessageSendRequest();
      const r1 = await MessageAPI.send(form1);
      const r2 = await MessageAPI.send(form2);
      createdMessageIds.push(...r1.messageIds, ...r2.messageIds);

      const result = await MessageAPI.markAllRead();
      expect(result.affectedCount).toBeGreaterThanOrEqual(2);

      const unread = await MessageAPI.getUnreadCount();
      expect(unread.count).toBe(0);
    });

    test("正向测试：按类型标记已读", async () => {
      const { messageIds } = await sendMessage({ type: "member" });
      createdMessageIds.push(...messageIds);

      const result = await MessageAPI.markAllRead("member");
      expect(result.affectedCount).toBeGreaterThanOrEqual(1);
    });
  });

  describe("DELETE /api/v1/messages/{ids} - 删除消息", () => {
    test("正向测试：单条删除", async () => {
      const { messageIds, id } = await sendMessage();

      await deleteMessages(messageIds);

      await expectBizError(MessageAPI.getDetail(id), [
        "A0550",
        "A0401",
        "A0400",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("正向测试：批量删除", async () => {
      const { messageIds: ids1 } = await sendMessage();
      const { messageIds: ids2 } = await sendMessage();

      await deleteMessages([ids1[0]!, ids2[0]!]);

      const result = await MessageAPI.getPage({ pageNum: 1, pageSize: 100 });
      const found = result.list.filter((m: MessageVO) => m.id === ids1[0] || m.id === ids2[0]);
      expect(found.length).toBe(0);
    });

    test("边界：删除不存在的消息幂等静默成功", async () => {
      // 后端软删除按 recipient 过滤，删除不存在的消息返回 00000（静默成功，不报错）
      await expect(MessageAPI.deleteByIds("999999999")).resolves.toBeUndefined();
    });

    test("边界：删除他人消息幂等静默成功（不暴露存在性）", async () => {
      // admin 发送给 user 的消息，admin 尝试删除——后端软删除按 recipient 过滤，
      // 删除他人消息返回 00000（静默成功），不暴露消息存在性
      const { messageIds } = await sendMessage({ recipientIds: [USERS.USER.id] });
      const msgId = messageIds[0]!;

      await expect(MessageAPI.deleteByIds(String(msgId))).resolves.toBeUndefined();
      // 切换 user 清理该消息
      await login(USERS.USER.username);
      await MessageAPI.deleteByIds(String(msgId));
      await login(USERS.ADMIN.username);
    });
  });

  describe("GET /api/v1/messages/search - 搜索消息", () => {
    test("正向测试：按标题关键字搜索", async () => {
      const uniqueTitle = "search_test_" + Date.now();
      const { messageIds } = await sendMessage({ title: uniqueTitle });

      const result = await MessageAPI.search({ keyword: uniqueTitle });
      expect(result.list.length).toBeGreaterThanOrEqual(1);
      expect(result.list.some((m: MessageVO) => m.id === messageIds[0])).toBe(true);

      await deleteMessages(messageIds);
    });

    test("正向测试：按正文关键字搜索", async () => {
      const uniqueContent = "content_search_" + Date.now();
      const { messageIds } = await sendMessage({ content: uniqueContent });

      const result = await MessageAPI.search({ keyword: uniqueContent });
      expect(result.list.length).toBeGreaterThanOrEqual(1);

      await deleteMessages(messageIds);
    });

    test("边界：无匹配结果返回空列表", async () => {
      const result = await MessageAPI.search({ keyword: "nonexistent_xyz_99999" });
      expect(result.list.length).toBe(0);
    });

    test("安全：特殊字符搜索不引发 XSS 风险", async () => {
      const result = await MessageAPI.search({ keyword: "<script>alert(1)</script>" });
      expect(Array.isArray(result.list)).toBe(true);
      const jsonStr = JSON.stringify(result);
      expect(jsonStr).not.toContain("<script>");
    });
  });

  describe("GET/PATCH /api/v1/notification-settings - 通知偏好设置", () => {
    test("正向测试：获取通知设置", async () => {
      const settings = await NotificationSettingAPI.get();
      expect(typeof settings.pushEnabled).toBe("boolean");
      expect(typeof settings.dndEnabled).toBe("boolean");
      expect(settings.dndStart).toBeDefined();
      expect(settings.dndEnd).toBeDefined();
      expect(settings.preferences).toBeDefined();
    });

    test("正向测试：修改推送开关和免打扰设置", async () => {
      await NotificationSettingAPI.update({
        pushEnabled: false,
        dndEnabled: true,
        dndStart: "23:00:00",
        dndEnd: "07:00:00",
      });

      const settings = await NotificationSettingAPI.get();
      expect(settings.pushEnabled).toBe(false);
      expect(settings.dndEnabled).toBe(true);
      expect(settings.dndStart).toBe("23:00:00");
      expect(settings.dndEnd).toBe("07:00:00");

      await NotificationSettingAPI.update({
        pushEnabled: true,
        dndEnabled: false,
        dndStart: "22:00:00",
        dndEnd: "08:00:00",
      });
    });

    test("正向测试：修改模块开关偏好", async () => {
      await NotificationSettingAPI.update({
        preferences: {
          moduleSwitches: { prediction: false, feedback: true, announcement: true },
        },
      });

      const settings = await NotificationSettingAPI.get();
      expect(settings.preferences.moduleSwitches.prediction).toBe(false);

      await NotificationSettingAPI.update({
        preferences: {
          moduleSwitches: { prediction: true, feedback: true, announcement: true },
        },
      });
    });

    test("边界：免打扰时间跨天保存成功", async () => {
      await NotificationSettingAPI.update({
        dndEnabled: true,
        dndStart: "22:00:00",
        dndEnd: "08:00:00",
      });

      const settings = await NotificationSettingAPI.get();
      expect(settings.dndEnabled).toBe(true);
      expect(settings.dndStart).toBe("22:00:00");
      expect(settings.dndEnd).toBe("08:00:00");

      // 恢复默认
      await NotificationSettingAPI.update({
        dndEnabled: false,
        dndStart: "22:00:00",
        dndEnd: "08:00:00",
      });
    });
  });

  describe("权限测试 - 普通用户管理操作应失败", () => {
    beforeAll(async () => {
      await login(USERS.USER.username);
    });

    afterAll(async () => {
      await login(USERS.ADMIN.username);
    });

    test("边界：普通用户创建公告应失败", async () => {
      const form = createAnnouncementForm();
      await expectBizError(AnnouncementAPI.create(form), [
        "A0301",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("边界：普通用户发送公告应失败", async () => {
      await expectBizError(AnnouncementAPI.send(1), ["A0301", "A0400", "B0001", "ERR_BAD_REQUEST"]);
    });

    test("边界：普通用户编辑消息模板应失败", async () => {
      await expectBizError(MessageTemplateAPI.update(1, { name: "test" } as any), [
        "A0301",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("公告管理接口", () => {
    test("正向测试：创建公告草稿", async () => {
      const form = createAnnouncementForm();
      const result = await AnnouncementAPI.create(form);
      expect(result.id).toBeDefined();
      testAnnouncementId = result.id;

      const detail = await AnnouncementAPI.getDetail(result.id);
      expect(detail.title).toBe(form.title);
      expect(detail.status).toBe(1);
    });

    test("正向测试：编辑草稿公告", async () => {
      const newTitle = "test_公告_edited_" + Date.now();
      await AnnouncementAPI.update(testAnnouncementId, { title: newTitle });

      const detail = await AnnouncementAPI.getDetail(testAnnouncementId);
      expect(detail.title).toBe(newTitle);
    });

    test("正向测试：发送公告", async () => {
      const result = await AnnouncementAPI.send(testAnnouncementId);
      expect(result.sentCount).toBeGreaterThanOrEqual(1);

      const detail = await AnnouncementAPI.getDetail(testAnnouncementId);
      expect(detail.status).toBe(3);
    });

    test("边界：编辑已发送公告应报错", async () => {
      await expectBizError(AnnouncementAPI.update(testAnnouncementId, { title: "should_fail" }), [
        "A0553",
        "A0502",
        "A0500",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("边界：取消已发送公告应报错", async () => {
      await expectBizError(AnnouncementAPI.cancel(testAnnouncementId), [
        "A0553",
        "A0502",
        "A0500",
        "ERR_BAD_REQUEST",
      ]);
    });

    // /api/v1/announcements 已加入防重排除列表，重复发送由业务状态校验拦截（A0553）
    test("边界：发送已发送公告应报错", async () => {
      await expectBizError(AnnouncementAPI.send(testAnnouncementId), ["A0553"]);
    });

    test("正向测试：创建定时公告并取消", async () => {
      const futureTime = new Date(Date.now() + 86400000)
        .toISOString()
        .replace("T", " ")
        .substring(0, 19);
      const form = createAnnouncementForm({ sendTime: futureTime });
      const { id } = await AnnouncementAPI.create(form);

      const detail = await AnnouncementAPI.getDetail(id);
      expect(detail.status).toBe(2);

      await AnnouncementAPI.cancel(id);
      const cancelled = await AnnouncementAPI.getDetail(id);
      expect(cancelled.status).toBe(4);

      await AnnouncementAPI.deleteById(id);
    });

    test("正向测试：公告分页列表", async () => {
      const result = await AnnouncementAPI.getPage({ pageNum: 1, pageSize: 10 });
      expect(Array.isArray(result.list)).toBe(true);
    });

    test("正向测试：按标题搜索公告", async () => {
      const result = await AnnouncementAPI.getPage({ title: "test_", pageNum: 1, pageSize: 10 });
      expect(result.list).toBeDefined();
    });

    test("正向测试：按状态筛选公告", async () => {
      const result = await AnnouncementAPI.getPage({ status: 3, pageNum: 1, pageSize: 10 });
      for (const a of result.list) {
        expect(a.status).toBe(3);
      }
    });

    test("参数校验：缺少type应失败", async () => {
      await expectBizError(
        AnnouncementAPI.create({ title: "test_missing_type", content: "内容" } as any),
        ["A0400", "B0001", "ERR_BAD_REQUEST"]
      );
    });

    test("正向测试：删除公告", async () => {
      await AnnouncementAPI.deleteById(testAnnouncementId);
      await expectBizError(AnnouncementAPI.getDetail(testAnnouncementId), [
        "A0552",
        "A0401",
        "A0400",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("参数校验：公告标题过短应报错", async () => {
      await expectBizError(AnnouncementAPI.create(createAnnouncementForm({ title: "a" })), [
        "A0400",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("消息模板管理接口", () => {
    test("正向测试：模板分页列表", async () => {
      const result = await MessageTemplateAPI.getPage({ pageNum: 1, pageSize: 20 });
      expect(Array.isArray(result.list)).toBe(true);
    });

    test("正向测试：模板详情", async () => {
      const page = await MessageTemplateAPI.getPage({ pageNum: 1, pageSize: 1 });
      if (page.list.length === 0) {
        console.warn("无消息模板数据，跳过模板详情验证");
        return;
      }

      const detail = await MessageTemplateAPI.getDetail(page.list[0]!.id);
      expect(detail.code).toBeDefined();
      expect(detail.name).toBeDefined();
      expect(detail.titleTemplate).toBeDefined();
    });

    test("正向测试：编辑模板", async () => {
      const page = await MessageTemplateAPI.getPage({ pageNum: 1, pageSize: 1 });
      if (page.list.length === 0) {
        console.warn("无消息模板数据，跳过模板编辑测试");
        return;
      }

      const template = page.list[0]!;
      const originalName = template.name;
      const newName = "test_edited_" + Date.now();

      await MessageTemplateAPI.update(template.id, { name: newName });
      const detail = await MessageTemplateAPI.getDetail(template.id);
      expect(detail.name).toBe(newName);

      await MessageTemplateAPI.update(template.id, { name: originalName });
    });

    test("边界：查询不存在的模板应失败", async () => {
      await expectBizError(MessageTemplateAPI.getDetail(99999999), [
        "A0401",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });
});
