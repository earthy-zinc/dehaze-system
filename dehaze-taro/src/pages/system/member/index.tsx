import React, { useState, useEffect, useCallback } from "react";
import { View, Text, ScrollView } from "@tarojs/components";
import Taro from "@tarojs/taro";
import { Tag, Loading, Empty, Popup, Input } from "@taroify/core";
import { MemberAPI } from "dehaze-sdk-js";
import type { MemberPageVO, BenefitVO, GrowthLogVO } from "dehaze-sdk-js";
import PageLayout from "@/layout";
import { usePermission } from "@/hooks/usePermission";
import { getErrorMessage } from "@/utils/error";
import "./index.less";

const LEVEL_LABELS: Record<string, string> = {
  level_0: "游客",
  level_1: "普通用户",
  level_2: "白银会员",
  level_3: "黄金会员",
};

const MemberManagePage: React.FC = () => {
  const { hasPermission } = usePermission();
  const canView = hasPermission("sys:member:*");
  const canAdjust = hasPermission("sys:member:edit");

  const [members, setMembers] = useState<MemberPageVO[]>([]);
  const [loading, setLoading] = useState(false);
  const [total, setTotal] = useState(0);
  const [pageNum, setPageNum] = useState(1);
  const [keyword, setKeyword] = useState("");
  const [levelFilter, setLevelFilter] = useState("");

  const [benefits, setBenefits] = useState<BenefitVO[]>([]);
  const [growthLogs, setGrowthLogs] = useState<GrowthLogVO[]>([]);
  const [selectedUserId, setSelectedUserId] = useState<number | null>(null);
  const [growthPopupVisible, setGrowthPopupVisible] = useState(false);
  const [levelPopupVisible, setLevelPopupVisible] = useState(false);
  const [adjustLevelCode, setAdjustLevelCode] = useState("");

  const fetchMembers = useCallback(
    async (page: number, kw: string, lv: string) => {
      setLoading(true);
      try {
        const params: any = { pageNum: page, pageSize: 15 };
        if (kw) params.keywords = kw;
        if (lv) params.levelCode = lv;
        const res = await MemberAPI.getPage(params);
        setMembers(res.list);
        setTotal(res.total);
        setPageNum(page);
      } catch (err: unknown) {
        Taro.showToast({
          title: getErrorMessage(err, "加载会员列表失败"),
          icon: "none",
        });
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const fetchBenefits = useCallback(async () => {
    try {
      const list = await MemberAPI.listBenefits();
      setBenefits(list || []);
    } catch {
      // 静默
    }
  }, []);

  useEffect(() => {
    fetchMembers(1, "", "");
    fetchBenefits();
  }, [fetchMembers, fetchBenefits]);

  const handleSearch = () => {
    fetchMembers(1, keyword, levelFilter);
  };

  const handleLoadMore = () => {
    if (members.length < total) {
      fetchMembers(pageNum + 1, keyword, levelFilter);
    }
  };

  const handleViewGrowth = async (userId: number) => {
    setSelectedUserId(userId);
    setGrowthPopupVisible(true);
    try {
      const res = await MemberAPI.getGrowthLogs({ pageNum: 1, pageSize: 20 });
      setGrowthLogs(res.list || []);
    } catch {
      setGrowthLogs([]);
    }
  };

  const handleToggleStatus = async (member: MemberPageVO) => {
    if (!canAdjust) return;
    const newStatus = member.status === 1 ? 0 : 1;
    const label = newStatus === 0 ? "冻结" : "解冻";
    const res = await Taro.showModal({
      title: `确认${label}`,
      content: `确定要${label}用户「${member.nickname || member.username}」吗？`,
    });
    if (!res.confirm) return;
    try {
      await MemberAPI.updateStatus(member.userId, { status: newStatus as any });
      Taro.showToast({ title: `${label}成功`, icon: "success" });
      fetchMembers(pageNum, keyword, levelFilter);
    } catch (err: unknown) {
      Taro.showToast({ title: getErrorMessage(err, "操作失败"), icon: "none" });
    }
  };

  const handleAdjustLevel = async () => {
    if (!selectedUserId || !adjustLevelCode) {
      Taro.showToast({ title: "请选择等级", icon: "none" });
      return;
    }
    try {
      await MemberAPI.adjustLevel(selectedUserId, {
        levelCode: adjustLevelCode as any,
        reason: "管理员手动调整",
      });
      Taro.showToast({ title: "等级调整成功", icon: "success" });
      setLevelPopupVisible(false);
      fetchMembers(pageNum, keyword, levelFilter);
    } catch (err: unknown) {
      Taro.showToast({ title: getErrorMessage(err, "调整失败"), icon: "none" });
    }
  };

  const openLevelAdjust = (userId: number) => {
    setSelectedUserId(userId);
    setAdjustLevelCode("");
    setLevelPopupVisible(true);
  };

  if (!canView) {
    return (
      <PageLayout level="L2" title="会员管理">
        <View className="no-permission">
          <Text>无权限访问</Text>
        </View>
      </PageLayout>
    );
  }

  return (
    <PageLayout level="L2" title="会员管理">
      <View className="system-manage-page">
        {/* 搜索栏 */}
        <View className="search-bar">
          <Input
            className="search-input"
            placeholder="搜索用户名/昵称"
            value={keyword}
            onInput={(e) => setKeyword(e.detail.value)}
            onConfirm={handleSearch}
          />
          <View className="filter-row">
            {["", "level_1", "level_2", "level_3"].map((lv) => (
              <Tag
                key={lv}
                color={levelFilter === lv ? "primary" : "default"}
                size="small"
                onClick={() => {
                  setLevelFilter(lv);
                  fetchMembers(1, keyword, lv);
                }}
              >
                {lv ? LEVEL_LABELS[lv] || lv : "全部"}
              </Tag>
            ))}
          </View>
        </View>

        {/* 会员列表 */}
        <ScrollView
          scrollY
          className="list-scroll"
          onScrollToLower={handleLoadMore}
        >
          {loading && members.length === 0 ? (
            <View className="loading-wrapper">
              <Loading>加载中...</Loading>
            </View>
          ) : members.length === 0 ? (
            <Empty>
              <Empty.Description>暂无会员数据</Empty.Description>
            </Empty>
          ) : (
            members.map((m) => (
              <View key={m.userId} className="list-card">
                <View className="card-header">
                  <View className="card-title-row">
                    <Text className="card-name">
                      {m.nickname || m.username}
                    </Text>
                    <Tag
                      size="small"
                      color={m.status === 1 ? "success" : "danger"}
                    >
                      {m.status === 1 ? "正常" : "冻结"}
                    </Tag>
                  </View>
                  <Text className="card-id">ID: {m.userId}</Text>
                </View>
                <View className="card-meta">
                  <Text className="meta-item">
                    等级:{" "}
                    {m.levelName || LEVEL_LABELS[m.levelCode] || m.levelCode}
                  </Text>
                  <Text className="meta-item">成长值: {m.growthValue}</Text>
                  <Text className="meta-item">已用: {m.monthlyUsed}</Text>
                  {m.expireTime && (
                    <Text className="meta-item">
                      到期: {new Date(m.expireTime).toLocaleDateString("zh-CN")}
                    </Text>
                  )}
                </View>
                {canAdjust && (
                  <View className="card-actions">
                    <View
                      className="action-btn"
                      onClick={() => handleViewGrowth(m.userId)}
                    >
                      成长记录
                    </View>
                    <View
                      className="action-btn"
                      onClick={() => openLevelAdjust(m.userId)}
                    >
                      调整等级
                    </View>
                    <View
                      className={`action-btn ${m.status === 1 ? "danger" : "primary"}`}
                      onClick={() => handleToggleStatus(m)}
                    >
                      {m.status === 1 ? "冻结" : "解冻"}
                    </View>
                  </View>
                )}
              </View>
            ))
          )}
          {members.length > 0 && members.length < total && (
            <View className="load-more" onClick={handleLoadMore}>
              <Text>加载更多</Text>
            </View>
          )}
        </ScrollView>

        {/* 成长值流水弹窗 */}
        <Popup
          open={growthPopupVisible}
          placement="bottom"
          rounded
          onClose={() => setGrowthPopupVisible(false)}
        >
          <View className="popup-content">
            <View className="popup-header">
              <Text className="popup-title">成长记录</Text>
              <Text
                className="popup-close"
                onClick={() => setGrowthPopupVisible(false)}
              >
                ×
              </Text>
            </View>
            <ScrollView
              scrollY
              className="popup-scroll"
              style={{ maxHeight: "60vh" }}
            >
              {growthLogs.length === 0 ? (
                <Empty>
                  <Empty.Description>暂无记录</Empty.Description>
                </Empty>
              ) : (
                growthLogs.map((log) => (
                  <View key={log.id} className="log-item">
                    <View className="log-left">
                      <Text className="log-reason">
                        {log.reason || "成长值变动"}
                      </Text>
                      <Text className="log-time">
                        {new Date(log.createTime).toLocaleString("zh-CN")}
                      </Text>
                    </View>
                    <Text
                      className={`log-value ${log.changeValue > 0 ? "positive" : "negative"}`}
                    >
                      {log.changeValue > 0 ? "+" : ""}
                      {log.changeValue}
                    </Text>
                  </View>
                ))
              )}
            </ScrollView>
          </View>
        </Popup>

        {/* 等级调整弹窗 */}
        <Popup
          open={levelPopupVisible}
          placement="bottom"
          rounded
          onClose={() => setLevelPopupVisible(false)}
        >
          <View className="popup-content">
            <View className="popup-header">
              <Text className="popup-title">调整等级</Text>
              <Text
                className="popup-close"
                onClick={() => setLevelPopupVisible(false)}
              >
                ×
              </Text>
            </View>
            <View className="popup-body">
              <View className="level-grid">
                {benefits.length > 0
                  ? benefits.map((b) => (
                      <View
                        key={b.levelCode}
                        className={`level-option ${adjustLevelCode === b.levelCode ? "selected" : ""}`}
                        onClick={() => setAdjustLevelCode(b.levelCode)}
                      >
                        <Text className="level-name">{b.levelName}</Text>
                        <Text className="level-growth">
                          {b.growthMin}-{b.growthMax}
                        </Text>
                      </View>
                    ))
                  : ["level_1", "level_2", "level_3"].map((lv) => (
                      <View
                        key={lv}
                        className={`level-option ${adjustLevelCode === lv ? "selected" : ""}`}
                        onClick={() => setAdjustLevelCode(lv)}
                      >
                        <Text className="level-name">{LEVEL_LABELS[lv]}</Text>
                      </View>
                    ))}
              </View>
              <View className="popup-confirm-btn" onClick={handleAdjustLevel}>
                <Text>确认调整</Text>
              </View>
            </View>
          </View>
        </Popup>
      </View>
    </PageLayout>
  );
};

export default MemberManagePage;
