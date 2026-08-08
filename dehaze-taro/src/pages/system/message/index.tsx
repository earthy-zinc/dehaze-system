import React, { useState, useEffect, useCallback } from "react";
import { View, Text, ScrollView } from "@tarojs/components";
import Taro from "@tarojs/taro";
import { Tag, Loading, Empty, Tabs, Popup, Input } from "@taroify/core";
import { AnnouncementAPI, MessageTemplateAPI } from "dehaze-sdk-js";
import type { AnnouncementVO, AnnouncementForm, MessageTemplateVO } from "dehaze-sdk-js";
import PageLayout from "@/layout";
import { usePermission } from "@/hooks/usePermission";
import { getErrorMessage } from "@/utils/error";
import "./index.less";

const STATUS_MAP: Record<number, { label: string; color: string }> = {
  1: { label: "草稿", color: "default" },
  2: { label: "待发送", color: "warning" },
  3: { label: "已发送", color: "success" },
  4: { label: "已取消", color: "default" },
};

const MessageManagePage: React.FC = () => {
  const { hasPermission } = usePermission();
  const canAddAnnouncement = hasPermission("notify:announcement:add");
  const canEditAnnouncement = hasPermission("notify:announcement:edit");
  const canDeleteAnnouncement = hasPermission("notify:announcement:delete");
  const canSendAnnouncement = hasPermission("notify:announcement:send");
  const canCancelAnnouncement = hasPermission("notify:announcement:cancel");
  const canManageTemplate = hasPermission("notify:template:edit");

  const canManageAnnouncement = hasPermission([
    "notify:announcement:add",
    "notify:announcement:edit",
    "notify:announcement:delete",
    "notify:announcement:send",
    "notify:announcement:cancel",
  ]);

  const [tab, setTab] = useState(0);

  const [announcements, setAnnouncements] = useState<AnnouncementVO[]>([]);
  const [annLoading, setAnnLoading] = useState(false);
  const [annTotal, setAnnTotal] = useState(0);
  const [annPageNum, setAnnPageNum] = useState(1);
  const [annKeyword, setAnnKeyword] = useState("");
  const [annStatusFilter, setAnnStatusFilter] = useState<number | undefined>();

  const [templates, setTemplates] = useState<MessageTemplateVO[]>([]);
  const [tplLoading, setTplLoading] = useState(false);
  const [tplTotal, setTplTotal] = useState(0);
  const [tplPageNum, setTplPageNum] = useState(1);

  const [annPopupVisible, setAnnPopupVisible] = useState(false);
  const [editingAnn, setEditingAnn] = useState<AnnouncementVO | null>(null);
  const [annForm, setAnnForm] = useState({
    title: "",
    content: "",
    type: "operation",
    importance: 1,
    targetScope: "all",
    sendTime: "",
    expireTime: "",
  });

  const [tplPopupVisible, setTplPopupVisible] = useState(false);
  const [editingTpl, setEditingTpl] = useState<MessageTemplateVO | null>(null);
  const [tplForm, setTplForm] = useState({
    name: "",
    titleTemplate: "",
    contentTemplate: "",
    priority: 1,
    status: 1,
  });

  const fetchAnnouncements = useCallback(
    async (page: number, kw: string, status?: number) => {
      setAnnLoading(true);
      try {
        const params: any = { pageNum: page, pageSize: 15 };
        if (kw) params.title = kw;
        if (status !== undefined) params.status = status;
        const res = await AnnouncementAPI.getPage(params);
        setAnnouncements(res.list);
        setAnnTotal(res.total);
        setAnnPageNum(page);
      } catch (err: unknown) {
        Taro.showToast({
          title: getErrorMessage(err, "加载公告失败"),
          icon: "none",
        });
      } finally {
        setAnnLoading(false);
      }
    },
    []
  );

  const fetchTemplates = useCallback(async (page: number) => {
    setTplLoading(true);
    try {
      const res = await MessageTemplateAPI.getPage({
        pageNum: page,
        pageSize: 15,
      });
      setTemplates(res.list);
      setTplTotal(res.total);
      setTplPageNum(page);
    } catch (err: unknown) {
      Taro.showToast({
        title: getErrorMessage(err, "加载模板失败"),
        icon: "none",
      });
    } finally {
      setTplLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAnnouncements(1, "", annStatusFilter);
    fetchTemplates(1);
  }, [fetchAnnouncements, fetchTemplates]);

  const handleAnnSearch = () => {
    fetchAnnouncements(1, annKeyword, annStatusFilter);
  };

  const handleLoadMoreAnn = () => {
    if (announcements.length < annTotal) {
      fetchAnnouncements(annPageNum + 1, annKeyword, annStatusFilter);
    }
  };

  const handleLoadMoreTpl = () => {
    if (templates.length < tplTotal) {
      fetchTemplates(tplPageNum + 1);
    }
  };

  const handleSendAnnouncement = async (id: number) => {
    if (!canSendAnnouncement) {
      Taro.showToast({ title: "无发送权限", icon: "none" });
      return;
    }
    try {
      const result = await AnnouncementAPI.send(id);
      Taro.showToast({
        title: `已发送给 ${result.sentCount} 人`,
        icon: "success",
      });
      fetchAnnouncements(annPageNum, annKeyword, annStatusFilter);
    } catch (err: unknown) {
      Taro.showToast({ title: getErrorMessage(err, "发送失败"), icon: "none" });
    }
  };

  const handleCancelAnnouncement = async (id: number) => {
    if (!canCancelAnnouncement) {
      Taro.showToast({ title: "无操作权限", icon: "none" });
      return;
    }
    try {
      await AnnouncementAPI.cancel(id);
      Taro.showToast({ title: "已取消", icon: "success" });
      fetchAnnouncements(annPageNum, annKeyword, annStatusFilter);
    } catch (err: unknown) {
      Taro.showToast({ title: getErrorMessage(err, "操作失败"), icon: "none" });
    }
  };

  const handleDeleteAnnouncement = async (id: number) => {
    if (!canDeleteAnnouncement) {
      Taro.showToast({ title: "无删除权限", icon: "none" });
      return;
    }
    const res = await Taro.showModal({
      title: "确认删除",
      content: "确定要删除这条公告吗？",
    });
    if (!res.confirm) return;
    try {
      await AnnouncementAPI.deleteById(id);
      Taro.showToast({ title: "已删除", icon: "success" });
      fetchAnnouncements(annPageNum, annKeyword, annStatusFilter);
    } catch (err: unknown) {
      Taro.showToast({ title: getErrorMessage(err, "删除失败"), icon: "none" });
    }
  };

  const openCreateAnn = () => {
    if (!canAddAnnouncement) {
      Taro.showToast({ title: "无创建权限", icon: "none" });
      return;
    }
    setEditingAnn(null);
    setAnnForm({
      title: "",
      content: "",
      type: "operation",
      importance: 1,
      targetScope: "all",
      sendTime: "",
      expireTime: "",
    });
    setAnnPopupVisible(true);
  };

  const openEditAnn = (a: AnnouncementVO) => {
    if (!canEditAnnouncement) {
      Taro.showToast({ title: "无编辑权限", icon: "none" });
      return;
    }
    if (a.status !== 1 && a.status !== 2) {
      Taro.showToast({ title: "仅草稿/待发送可编辑", icon: "none" });
      return;
    }
    setEditingAnn(a);
    setAnnForm({
      title: a.title,
      content: a.content || "",
      type: a.type,
      importance: a.importance,
      targetScope: a.targetScope,
      sendTime: a.sendTime || "",
      expireTime: a.expireTime || "",
    });
    setAnnPopupVisible(true);
  };

  const handleSaveAnn = async () => {
    if (!annForm.title.trim()) {
      Taro.showToast({ title: "请输入公告标题", icon: "none" });
      return;
    }
    try {
      const data: AnnouncementForm = {
        title: annForm.title,
        content: annForm.content,
        type: annForm.type,
        importance: annForm.importance,
        targetScope: annForm.targetScope,
      };
      if (annForm.sendTime) data.sendTime = annForm.sendTime;
      if (annForm.expireTime) data.expireTime = annForm.expireTime;

      if (editingAnn) {
        await AnnouncementAPI.update(editingAnn.id, data);
        Taro.showToast({ title: "更新成功", icon: "success" });
      } else {
        await AnnouncementAPI.create(data);
        Taro.showToast({ title: "创建成功", icon: "success" });
      }
      setAnnPopupVisible(false);
      fetchAnnouncements(1, annKeyword, annStatusFilter);
    } catch (err: unknown) {
      Taro.showToast({ title: getErrorMessage(err, "操作失败"), icon: "none" });
    }
  };

  const openEditTpl = (tpl: MessageTemplateVO) => {
    if (!canManageTemplate) {
      Taro.showToast({ title: "无编辑权限", icon: "none" });
      return;
    }
    setEditingTpl(tpl);
    setTplForm({
      name: tpl.name,
      titleTemplate: tpl.titleTemplate,
      contentTemplate: tpl.contentTemplate || "",
      priority: tpl.priority,
      status: tpl.status,
    });
    setTplPopupVisible(true);
  };

  const handleSaveTpl = async () => {
    if (!editingTpl) return;
    try {
      await MessageTemplateAPI.update(editingTpl.id, {
        name: tplForm.name,
        titleTemplate: tplForm.titleTemplate,
        contentTemplate: tplForm.contentTemplate,
        priority: tplForm.priority,
        status: tplForm.status,
      });
      Taro.showToast({ title: "保存成功", icon: "success" });
      setTplPopupVisible(false);
      fetchTemplates(tplPageNum);
    } catch (err: unknown) {
      Taro.showToast({ title: getErrorMessage(err, "保存失败"), icon: "none" });
    }
  };

  const getStatusInfo = (status: number) => {
    return STATUS_MAP[status] || { label: `未知(${status})`, color: "default" };
  };

  return (
    <PageLayout level="L2" title="消息管理">
      <View className="system-manage-page">
        <Tabs value={tab} onChange={setTab}>
          <Tabs.TabPane title="公告管理" />
          <Tabs.TabPane title="消息模板" />
        </Tabs>

        {tab === 0 && canManageAnnouncement && (
          <>
            <View className="search-bar">
              <View className="search-row">
                <Input
                  className="search-input"
                  placeholder="搜索公告标题"
                  value={annKeyword}
                  onInput={(e) => setAnnKeyword(e.detail.value)}
                  onConfirm={handleAnnSearch}
                />
                {canAddAnnouncement && (
                  <View className="create-btn" onClick={openCreateAnn}>
                    <Text>新建</Text>
                  </View>
                )}
              </View>
              <View className="status-filter-row">
                {[undefined, 1, 2, 3, 4].map((s) => (
                  <View
                    key={s ?? "all"}
                    className={`status-filter-item ${annStatusFilter === s ? "active" : ""}`}
                    onClick={() => {
                      setAnnStatusFilter(s);
                      fetchAnnouncements(1, annKeyword, s);
                    }}
                  >
                    <Text>
                      {s === undefined ? "全部" : STATUS_MAP[s]?.label || s}
                    </Text>
                  </View>
                ))}
              </View>
            </View>

            <ScrollView
              scrollY
              className="list-scroll"
              onScrollToLower={handleLoadMoreAnn}
            >
              {annLoading && announcements.length === 0 ? (
                <View className="loading-wrapper">
                  <Loading>加载中...</Loading>
                </View>
              ) : announcements.length === 0 ? (
                <Empty>
                  <Empty.Description>暂无公告</Empty.Description>
                </Empty>
              ) : (
                announcements.map((a) => {
                  const si = getStatusInfo(a.status);
                  return (
                    <View key={a.id} className="list-card">
                      <View className="card-header">
                        <View className="card-title-row">
                          <Text className="card-name">{a.title}</Text>
                          <Tag size="small" color={si.color as any}>
                            {a.statusLabel || si.label}
                          </Tag>
                        </View>
                        <Tag size="small" color="default">
                          {a.typeLabel || a.type}
                        </Tag>
                      </View>
                      <View className="card-meta">
                        <Text className="meta-item">
                          目标: {a.targetScopeLabel || a.targetScope}
                        </Text>
                        <Text className="meta-item">
                          重要度: {a.importanceLabel || a.importance}
                        </Text>
                        {a.sentCount !== undefined && (
                          <Text className="meta-item">发送: {a.sentCount}人</Text>
                        )}
                      </View>
                      <View className="card-meta">
                        <Text className="meta-item">
                          {new Date(a.createTime).toLocaleString("zh-CN")}
                        </Text>
                      </View>
                      <View className="card-actions">
                        {(a.status === 1 || a.status === 2) &&
                          canEditAnnouncement && (
                            <View
                              className="action-btn"
                              onClick={() => openEditAnn(a)}
                            >
                              编辑
                            </View>
                          )}
                        {(a.status === 1 || a.status === 2) &&
                          canSendAnnouncement && (
                            <View
                              className="action-btn primary"
                              onClick={() => handleSendAnnouncement(a.id)}
                            >
                              发送
                            </View>
                          )}
                        {a.status === 2 && canCancelAnnouncement && (
                          <View
                            className="action-btn warning"
                            onClick={() => handleCancelAnnouncement(a.id)}
                          >
                            取消
                          </View>
                        )}
                        {canDeleteAnnouncement && (
                          <View
                            className="action-btn danger"
                            onClick={() => handleDeleteAnnouncement(a.id)}
                          >
                            删除
                          </View>
                        )}
                      </View>
                    </View>
                  );
                })
              )}
              {announcements.length > 0 && announcements.length < annTotal && (
                <View className="load-more" onClick={handleLoadMoreAnn}>
                  <Text>加载更多</Text>
                </View>
              )}
            </ScrollView>
          </>
        )}

        {tab === 1 && (
          <ScrollView
            scrollY
            className="list-scroll"
            onScrollToLower={handleLoadMoreTpl}
          >
            {tplLoading && templates.length === 0 ? (
              <View className="loading-wrapper">
                <Loading>加载中...</Loading>
              </View>
            ) : templates.length === 0 ? (
              <Empty>
                <Empty.Description>暂无消息模板</Empty.Description>
              </Empty>
            ) : (
              templates.map((t) => (
                <View key={t.id} className="list-card">
                  <View className="card-header">
                    <View className="card-title-row">
                      <Text className="card-name">{t.name}</Text>
                      <Tag
                        size="small"
                        color={t.status === 1 ? "success" : "default"}
                      >
                        {t.status === 1 ? "启用" : "禁用"}
                      </Tag>
                    </View>
                    <Text className="card-id">{t.code}</Text>
                  </View>
                  <View className="card-meta">
                    <Text className="meta-item">类型: {t.type}</Text>
                    <Text className="meta-item">优先级: {t.priority}</Text>
                  </View>
                  <Text className="card-content" numberOfLines={1}>
                    标题模板: {t.titleTemplate}
                  </Text>
                  {t.variables && t.variables.length > 0 && (
                    <View className="card-meta">
                      <Text className="meta-item">
                        变量: {t.variables.map((v) => `{${v.name}}`).join(" ")}
                      </Text>
                    </View>
                  )}
                  <View className="card-actions">
                    {canManageTemplate && (
                      <View className="action-btn" onClick={() => openEditTpl(t)}>
                        编辑
                      </View>
                    )}
                  </View>
                </View>
              ))
            )}
            {templates.length > 0 && templates.length < tplTotal && (
              <View className="load-more" onClick={handleLoadMoreTpl}>
                <Text>加载更多</Text>
              </View>
            )}
          </ScrollView>
        )}

        {tab === 0 && !canManageAnnouncement && (
          <View className="no-permission">
            <Text>无公告管理权限</Text>
          </View>
        )}

        {/* 公告编辑弹窗 */}
        <Popup
          open={annPopupVisible}
          placement="bottom"
          rounded
          onClose={() => setAnnPopupVisible(false)}
        >
          <View className="popup-content">
            <View className="popup-header">
              <Text className="popup-title">
                {editingAnn ? "编辑公告" : "新建公告"}
              </Text>
              <Text
                className="popup-close"
                onClick={() => setAnnPopupVisible(false)}
              >
                ×
              </Text>
            </View>
            <View className="popup-body">
              <View className="form-item">
                <Text className="form-label">标题 *</Text>
                <Input
                  className="form-input"
                  placeholder="公告标题"
                  value={annForm.title}
                  onInput={(e) =>
                    setAnnForm({ ...annForm, title: e.detail.value })
                  }
                />
              </View>
              <View className="form-item">
                <Text className="form-label">内容</Text>
                <Input
                  className="form-input"
                  placeholder="公告内容"
                  value={annForm.content}
                  onInput={(e) =>
                    setAnnForm({ ...annForm, content: e.detail.value })
                  }
                />
              </View>
              <View className="popup-confirm-btn" onClick={handleSaveAnn}>
                <Text>{editingAnn ? "保存" : "创建"}</Text>
              </View>
            </View>
          </View>
        </Popup>

        {/* 模板编辑弹窗 */}
        <Popup
          open={tplPopupVisible}
          placement="bottom"
          rounded
          onClose={() => setTplPopupVisible(false)}
        >
          <View className="popup-content">
            <View className="popup-header">
              <Text className="popup-title">编辑模板</Text>
              <Text
                className="popup-close"
                onClick={() => setTplPopupVisible(false)}
              >
                ×
              </Text>
            </View>
            <View className="popup-body">
              <View className="form-item">
                <Text className="form-label">模板名称</Text>
                <Input
                  className="form-input"
                  value={tplForm.name}
                  onInput={(e) =>
                    setTplForm({ ...tplForm, name: e.detail.value })
                  }
                />
              </View>
              <View className="form-item">
                <Text className="form-label">标题模板</Text>
                <Input
                  className="form-input"
                  value={tplForm.titleTemplate}
                  onInput={(e) =>
                    setTplForm({ ...tplForm, titleTemplate: e.detail.value })
                  }
                />
              </View>
              <View className="form-item">
                <Text className="form-label">内容模板</Text>
                <Input
                  className="form-input"
                  value={tplForm.contentTemplate}
                  onInput={(e) =>
                    setTplForm({ ...tplForm, contentTemplate: e.detail.value })
                  }
                />
              </View>
              <View className="popup-confirm-btn" onClick={handleSaveTpl}>
                <Text>保存</Text>
              </View>
            </View>
          </View>
        </Popup>
      </View>
    </PageLayout>
  );
};

export default MessageManagePage;
