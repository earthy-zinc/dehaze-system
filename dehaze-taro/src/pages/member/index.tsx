import React, { useState, useEffect, useCallback } from "react";
import { View, Text, ScrollView } from "@tarojs/components";
import { Gift, Star } from "@taroify/icons";
import { MemberAPI } from "dehaze-sdk-js";
import type { MemberProfileVO, GrowthLogVO } from "dehaze-sdk-js";
import PageLayout from "@/layout";
import StatusTag from "@/components/common/StatusTag";
import "./index.less";

// 会员等级配置
const LEVEL_CONFIG = {
  1: { name: "普通用户", color: "#6b7280", icon: "👤" },
  2: { name: "白银会员", color: "#c0c0c0", icon: "🥈" },
  3: { name: "黄金会员", color: "#f59e0b", icon: "🥇" },
  4: { name: "钻石会员", color: "#3b82f6", icon: "💎" },
  5: { name: "至尊会员", color: "#7c3aed", icon: "👑" },
};

// 会员权益列表
const MEMBER_BENEFITS = [
  { icon: "⚡", title: "去雾次数", desc: "每日无限次去雾处理" },
  { icon: "🎯", title: "算法访问", desc: "解锁全部高级算法" },
  { icon: "💾", title: "存储空间", desc: "100GB 云端存储" },
  { icon: "🚀", title: "处理速度", desc: "优先队列，快速出图" },
  { icon: "👥", title: "团队协作", desc: "支持多人项目协作" },
  { icon: "🔌", title: "API 接口", desc: "开放 API 调用权限" },
];

const MemberPage: React.FC = () => {
  const [member, setMember] = useState<MemberProfileVO | null>(null);
  const [growthLogs, setGrowthLogs] = useState<GrowthLogVO[]>([]);

  // 加载会员信息
  const loadMemberInfo = useCallback(async () => {
    try {
      const info = await MemberAPI.getProfile();
      setMember(info);

      // 加载成长值记录
      try {
        const logsRes = await MemberAPI.getGrowthLogs({
          pageNum: 1,
          pageSize: 10,
        });
        setGrowthLogs((logsRes.list as unknown as GrowthLogVO[]) || []);
      } catch {
        // 静默失败
      }
    } catch {
      // 使用默认数据
      setMember({
        userId: 1,
        username: "",
        nickname: "用户",
        levelCode: "level_1",
        levelName: "普通用户",
        growthValue: 0,
        progressPercent: 0,
        monthlyDehazeQuota: 0,
        monthlyDehazeUsed: 0,
        monthlyEvaluateQuota: 0,
        monthlyEvaluateUsed: 0,
        benefits: {
          levelCode: "level_1",
          levelName: "普通用户",
          growthMin: 0,
          growthMax: 100,
          monthlyDehazeQuota: 0,
          monthlyEvaluateQuota: 0,
          historyRetention: 0,
          batchLimit: 0,
          priority: 0,
          advancedParams: 0,
          hdExport: 0,
          reportExport: 0,
          batchDownload: 0,
          sort: 0,
          status: 0,
        },
        status: 1,
      });
    }
  }, []);

  useEffect(() => {
    loadMemberInfo();
  }, [loadMemberInfo]);

  // 当前会员等级（从 levelCode 映射）
  const levelMap: Record<
    string,
    { name: string; color: string; icon: string }
  > = {
    ...LEVEL_CONFIG,
    level_0: { name: "未登录", color: "#6b7280", icon: "👤" },
    level_1: { name: "普通用户", color: "#6b7280", icon: "👤" },
    level_2: { name: "白银会员", color: "#c0c0c0", icon: "🥈" },
    level_3: { name: "黄金会员", color: "#f59e0b", icon: "🥇" },
  };
  const levelCfg = levelMap[member?.levelCode || "level_1"] || levelMap.level_1;

  // 成长值进度（API 已返回 progressPercent）
  const growthValue = member?.growthValue || 0;
  const progressPercent = member?.progressPercent || 0;
  const maxGrowthValue = 1000;
  const nextLevelGrowth = Math.max(0, maxGrowthValue - growthValue);

  return (
    <PageLayout showTabbar currentRoute="/pages/member/index" title="会员中心">
      <View className="member-page">
        <ScrollView
          scrollY
          className="member-scroll"
          enhanced
          showScrollbar={false}
        >
          {/* 会员头部卡片 */}
          <View className="member-header-card">
            <View className="header-bg" />
            <View className="header-content">
              {/* 头像和等级 */}
              <View className="member-avatar-section">
                <View className="avatar-wrapper">
                  <Text className="avatar-icon">{levelCfg.icon}</Text>
                </View>
                <View className="member-info-section">
                  <View className="member-name-row">
                    <Text className="member-level-name">{levelCfg.name}</Text>
                    <StatusTag status={member?.status ?? 1} size="small" />
                  </View>
                  <Text className="member-id">ID: {member?.userId || "-"}</Text>
                </View>
              </View>

              {/* 成长值进度条 */}
              <View className="growth-section">
                <View className="growth-header">
                  <View className="growth-left">
                    <Star size="14" color="#f59e0b" />
                    <Text className="growth-label">成长值</Text>
                  </View>
                  <Text className="growth-value">
                    {growthValue} / {maxGrowthValue}
                  </Text>
                </View>
                <View className="growth-bar-bg">
                  <View
                    className="growth-bar-fill"
                    style={{ width: `${progressPercent}%` }}
                  />
                </View>
                <Text className="growth-tip">
                  距下一级还需 {nextLevelGrowth} 成长值
                </Text>
              </View>
            </View>
          </View>

          {/* 会员权益 */}
          <View className="benefits-section">
            <View className="section-header">
              <Gift size="16" color="#3b82f6" />
              <Text className="section-title">会员权益</Text>
            </View>
            <View className="benefits-grid">
              {MEMBER_BENEFITS.map((benefit, idx) => (
                <View key={idx} className="benefit-card">
                  <Text className="benefit-icon">{benefit.icon}</Text>
                  <Text className="benefit-title">{benefit.title}</Text>
                  <Text className="benefit-desc">{benefit.desc}</Text>
                </View>
              ))}
            </View>
          </View>

          {/* 当前权益状态 */}
          <View className="current-plan-section">
            <View className="section-header">
              <Star size="16" color="#f59e0b" />
              <Text className="section-title">当前套餐</Text>
            </View>
            <View className="plan-card">
              <View className="plan-name">
                <Text className="plan-title">{levelCfg.name}</Text>
              </View>
              <View className="plan-badges">
                <StatusTag status={member?.status ?? 1} size="small" />
              </View>
              <View className="plan-details">
                <View className="plan-detail-item">
                  <Text className="detail-label">账号状态</Text>
                  <Text className="detail-value">
                    {member?.status === 1 ? "正常" : "冻结"}
                  </Text>
                </View>
              </View>
            </View>
          </View>

          {/* 成长值记录 */}
          {growthLogs.length > 0 && (
            <View className="growth-log-section">
              <View className="section-header">
                <Star size="16" color="#f59e0b" />
                <Text className="section-title">成长记录</Text>
              </View>
              <View className="growth-log-list">
                {growthLogs.slice(0, 5).map((log) => (
                  <View key={log.id} className="log-item">
                    <View className="log-icon">
                      <Star size="12" color="#f59e0b" />
                    </View>
                    <View className="log-content">
                      <Text className="log-desc">
                        {log.reason || "获得成长值"}
                      </Text>
                      <Text className="log-time">
                        {new Date(log.createTime).toLocaleString()}
                      </Text>
                    </View>
                    <Text className="log-value positive">
                      +{log.changeValue}
                    </Text>
                  </View>
                ))}
              </View>
            </View>
          )}

          {/* 页脚 */}
          <View className="member-footer">
            <Text className="footer-text">图像去雾系统 v1.0</Text>
          </View>
        </ScrollView>
      </View>
    </PageLayout>
  );
};

export default MemberPage;
