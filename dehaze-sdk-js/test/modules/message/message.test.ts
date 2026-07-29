import {
  AnnouncementAPI,
  MessageAPI,
  MessageTemplateAPI,
  NotificationSettingAPI,
  MessageVO,
} from "../../../index";
import { expectBizError } from "#/utils/assertion";
import { createAnnouncementForm, createMessageSendRequest } from "#/factories/message";

describe("消息通知模块接口测试", () => {
  let testMessageId: number;
  let testAnnouncementId: number;

  describe("POST /api/v1/messages/send - 内部消息发送", () => {
    test("正向测试：不使用模板发送消息", async () => {
      const form = createMessageSendRequest();
      const result = await MessageAPI.send(form);
      expect(result).toBeDefined();
      expect(result.messageIds).toBeDefined();
      expect(result.messageIds.length).toBe(1);
      testMessageId = result.messageIds[0]!;
    });

    test("正向测试：批量发送给多个接收人", async () => {
      const form = createMessageSendRequest({ recipientIds: [1, 2] });
      const result = await MessageAPI.send(form);
      expect(result.messageIds.length).toBe(2);
      await MessageAPI.deleteByIds(result.messageIds.join(","));
    });

    test("正向测试：幂等去重，相同bizModule+bizId不重复生成", async () => {
      const bizId = "idempotent_test_" + Date.now();
      const form = createMessageSendRequest({ bizModule: "test", bizId });
      const first = await MessageAPI.send(form);
      const second = await MessageAPI.send(form);
      expect(second.messageIds).toEqual(first.messageIds);
      await MessageAPI.deleteByIds(first.messageIds.join(","));
    });

    test("参数校验：缺少type应抛出业务错误", async () => {
      await expectBizError(
        MessageAPI.send({ recipientIds: [1], title: "x", content: "y" } as any),
        ["A0400", "A0410", "ERR_BAD_REQUEST"],
        undefined,
        true
      );
    });

    test("参数校验：空接收人列表应抛出业务错误", async () => {
      await expectBizError(
        MessageAPI.send(createMessageSendRequest({ recipientIds: [] })),
        ["A0400", "A0410", "ERR_BAD_REQUEST"],
        undefined,
        true
      );
    });
  });

  describe("GET /api/v1/messages - 消息列表", () => {
    test("正向测试：分页查询消息列表", async () => {
      const result = await MessageAPI.getPage({ pageNum: 1, pageSize: 20 });
      expect(result).toBeDefined();
      expect(result.list).toBeDefined();
      expect(Array.isArray(result.list)).toBe(true);
      expect(result.total).toBeGreaterThanOrEqual(0);
    });

    test("正向测试：按类型筛选消息", async () => {
      const result = await MessageAPI.getPage({ type: "business", pageNum: 1, pageSize: 10 });
      expect(result.list).toBeDefined();
      for (const msg of result.list) {
        expect(msg.type).toBe("business");
      }
    });

    test("正向测试：按已读状态筛选未读消息", async () => {
      const result = await MessageAPI.getPage({ readStatus: 0, pageNum: 1, pageSize: 10 });
      expect(result.list).toBeDefined();
      for (const msg of result.list) {
        expect(msg.readStatus).toBe(0);
      }
    });
  });

  describe("GET /api/v1/messages/unread-count - 未读消息数", () => {
    test("正向测试：查询未读消息数", async () => {
      const result = await MessageAPI.getUnreadCount();
      expect(result).toBeDefined();
      expect(typeof result.count).toBe("number");
      expect(result.count).toBeGreaterThanOrEqual(0);
    });
  });

  describe("GET /api/v1/messages/{id} - 消息详情", () => {
    test("正向测试：查看消息详情不自动标记已读", async () => {
      const form = createMessageSendRequest();
      const { messageIds } = await MessageAPI.send(form);
      const id = messageIds[0]!;

      const detail = await MessageAPI.getDetail(id);
      expect(detail).toBeDefined();
      expect(detail.id).toBe(id);
      expect(detail.title).toBe(form.title);
      expect(detail.content).toBe(form.content);
      expect(detail.readStatus).toBe(0);

      await MessageAPI.markRead(id);
      const readDetail = await MessageAPI.getDetail(id);
      expect(readDetail.readStatus).toBe(1);
      expect(readDetail.readTime).toBeDefined();

      await MessageAPI.deleteByIds(String(id));
    });

    test("异常：查看不存在的消息", async () => {
      await expectBizError(
        MessageAPI.getDetail(999999999),
        ["A0550", "A0401", "A0400", "ERR_BAD_REQUEST"],
        undefined,
        true
      );
    });
  });

  describe("PATCH /api/v1/messages/{id}/_read - 标记单条已读", () => {
    test("正向测试：标记未读消息为已读", async () => {
      const form = createMessageSendRequest();
      const { messageIds } = await MessageAPI.send(form);
      const id = messageIds[0]!;

      await MessageAPI.markRead(id);
      const detail = await MessageAPI.getDetail(id);
      expect(detail.readStatus).toBe(1);

      await MessageAPI.deleteByIds(String(id));
    });

    test("边界：重复标记已读幂等返回成功", async () => {
      const form = createMessageSendRequest();
      const { messageIds } = await MessageAPI.send(form);
      const id = messageIds[0]!;

      await MessageAPI.markRead(id);
      await MessageAPI.markRead(id);
      const detail = await MessageAPI.getDetail(id);
      expect(detail.readStatus).toBe(1);

      await MessageAPI.deleteByIds(String(id));
    });
  });

  describe("PATCH /api/v1/messages/_read-all - 全部标记已读", () => {
    test("正向测试：全部标记已读并返回受影响条数", async () => {
      const form1 = createMessageSendRequest();
      const form2 = createMessageSendRequest();
      await MessageAPI.send(form1);
      await MessageAPI.send(form2);

      const result = await MessageAPI.markAllRead();
      expect(result).toBeDefined();
      expect(result.affectedCount).toBeGreaterThanOrEqual(2);

      const unread = await MessageAPI.getUnreadCount();
      expect(unread.count).toBe(0);
    });

    test("正向测试：按类型标记已读", async () => {
      const form = createMessageSendRequest({ type: "member" });
      await MessageAPI.send(form);

      const result = await MessageAPI.markAllRead("member");
      expect(result.affectedCount).toBeGreaterThanOrEqual(1);
    });
  });

  describe("DELETE /api/v1/messages/{ids} - 删除消息", () => {
    test("正向测试：单条删除", async () => {
      const form = createMessageSendRequest();
      const { messageIds } = await MessageAPI.send(form);
      const id = messageIds[0]!;

      await MessageAPI.deleteByIds(String(id));

      await expectBizError(
        MessageAPI.getDetail(id),
        ["A0550", "A0401", "A0400", "ERR_BAD_REQUEST"],
        undefined,
        true
      );
    });

    test("正向测试：批量删除", async () => {
      const form1 = createMessageSendRequest();
      const form2 = createMessageSendRequest();
      const r1 = await MessageAPI.send(form1);
      const r2 = await MessageAPI.send(form2);
      const ids = r1.messageIds[0] + "," + r2.messageIds[0];

      await MessageAPI.deleteByIds(ids);

      const result = await MessageAPI.getPage({ pageNum: 1, pageSize: 100 });
      const found = result.list.filter(
        (m: MessageVO) => m.id === r1.messageIds[0] || m.id === r2.messageIds[0]
      );
      expect(found.length).toBe(0);
    });
  });

  describe("GET /api/v1/messages/search - 搜索消息", () => {
    test("正向测试：按标题关键字搜索", async () => {
      const uniqueTitle = "search_test_" + Date.now();
      const form = createMessageSendRequest({ title: uniqueTitle });
      const { messageIds } = await MessageAPI.send(form);

      const result = await MessageAPI.search({ keyword: uniqueTitle });
      expect(result.list.length).toBeGreaterThanOrEqual(1);
      expect(result.list.some((m: MessageVO) => m.id === messageIds[0])).toBe(true);

      await MessageAPI.deleteByIds(String(messageIds[0]));
    });
  });

  describe("GET/PATCH /api/v1/notification-settings - 通知偏好设置", () => {
    test("正向测试：获取通知设置", async () => {
      const settings = await NotificationSettingAPI.get();
      expect(settings).toBeDefined();
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
  });

  describe("公告管理接口", () => {
    test("正向测试：创建公告草稿", async () => {
      const form = createAnnouncementForm();
      const result = await AnnouncementAPI.create(form);
      expect(result).toBeDefined();
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
      expect(result).toBeDefined();
      expect(result.sentCount).toBeGreaterThanOrEqual(1);

      const detail = await AnnouncementAPI.getDetail(testAnnouncementId);
      expect(detail.status).toBe(3);
    });

    test("边界：编辑已发送公告应报错", async () => {
      await expectBizError(
        AnnouncementAPI.update(testAnnouncementId, { title: "should_fail" }),
        ["A0553", "A0502", "A0500", "ERR_BAD_REQUEST"],
        undefined,
        true
      );
    });

    test("边界：取消已发送公告应报错", async () => {
      await expectBizError(
        AnnouncementAPI.cancel(testAnnouncementId),
        ["A0553", "A0502", "A0500", "ERR_BAD_REQUEST"],
        undefined,
        true
      );
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
      expect(result).toBeDefined();
      expect(result.list).toBeDefined();
      expect(Array.isArray(result.list)).toBe(true);
    });

    test("正向测试：按标题搜索公告", async () => {
      const result = await AnnouncementAPI.getPage({ title: "test_", pageNum: 1, pageSize: 10 });
      expect(result.list).toBeDefined();
    });

    test("正向测试：删除公告", async () => {
      await AnnouncementAPI.deleteById(testAnnouncementId);
      await expectBizError(
        AnnouncementAPI.getDetail(testAnnouncementId),
        ["A0552", "A0401", "A0400", "ERR_BAD_REQUEST"],
        undefined,
        true
      );
    });

    test("参数校验：公告标题过短应报错", async () => {
      await expectBizError(
        AnnouncementAPI.create(createAnnouncementForm({ title: "a" })),
        ["A0400", "ERR_BAD_REQUEST"],
        undefined,
        true
      );
    });
  });

  describe("消息模板管理接口", () => {
    test("正向测试：模板分页列表", async () => {
      const result = await MessageTemplateAPI.getPage({ pageNum: 1, pageSize: 20 });
      expect(result).toBeDefined();
      expect(result.list).toBeDefined();
      expect(Array.isArray(result.list)).toBe(true);
    });

    test("正向测试：模板详情", async () => {
      const page = await MessageTemplateAPI.getPage({ pageNum: 1, pageSize: 1 });
      if (page.list.length === 0) return;

      const detail = await MessageTemplateAPI.getDetail(page.list[0]!.id);
      expect(detail).toBeDefined();
      expect(detail.code).toBeDefined();
      expect(detail.name).toBeDefined();
      expect(detail.titleTemplate).toBeDefined();
    });

    test("正向测试：编辑模板", async () => {
      const page = await MessageTemplateAPI.getPage({ pageNum: 1, pageSize: 1 });
      if (page.list.length === 0) return;

      const template = page.list[0]!;
      const originalName = template.name;
      const newName = "test_edited_" + Date.now();

      await MessageTemplateAPI.update(template.id, { name: newName });
      const detail = await MessageTemplateAPI.getDetail(template.id);
      expect(detail.name).toBe(newName);

      await MessageTemplateAPI.update(template.id, { name: originalName });
    });
  });
});
