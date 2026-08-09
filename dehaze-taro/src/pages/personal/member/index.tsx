import React, { useState, useEffect, useCallback } from "react";
import { View, Text, ScrollView } from "@tarojs/components";
import Taro from "@tarojs/taro";
import { Star, Gift } from "@taroify/icons";
import { MemberAPI } from "dehaze-sdk-js";
import type { MemberProfileVO, GrowthLogVO, BenefitVO } from "dehaze-sdk-js";
import PageLayout from "@/layout";
import "./index.less";

const LEVEL_CONFIG: Record<string, { name: string; color: string; icon: string }> = {
  level_0: { name: "游客", color: "#9ca3af", icon: "👤" },
  level_1: { name: "普通用户", color: "#6b7280", icon: "👤" },
  level_2: { name: "白银会员", color: "#c0c0c0", icon: "🥈" },
  level_3: { name: "黄金会员", color: "#f59e0b", icon: "🥇" },
};

const BENEFIT_LABELS: Record<string, string> = {
  monthlyDehazeQuota: "每月去雾次数",
  monthlyEvaluateQuota: "每月评估次数",
  historyRetention: "历史保留(天)",
  batchLimit: "批量上限",
  priority: "优先级",
  advancedParams: "高级参数",
  hdExport: "高清导出",
  reportExport: "报告导出",
  batchDownload: "批量下载",
};

const MemberPage: React.FC = () => {
  const [member, setMember] = useState<MemberProfileVO | null>(null);
  const [growthLogs, setGrowthLogs] = useState<GrowthLogVO[]>([]);

  const loadMemberInfo = useCallback(async () => {
    try {
      const info = await MemberAPI.getProfile();
      setMember(info);
      try {
        const logsRes = await MemberAPI.getGrowthLogs({ pageNum: 1, pageSize: 10 });
        setGrowthLogs(logsRes.list || []);
      } catch {
        // 静默
      }
    } catch {
      // 静默
    }
  }, []);

  useEffect(() => {
    loadMemberInfo();
  }, [loadMemberInfo]);

  const levelCfg = LEVEL_CONFIG[member?.levelCode || "level_1"] || LEVEL_CONFIG.level_1;
  const growthValue = member?.growthValue || 0;
  const progressPercent = member?.progressPercent || 0;
  const nextLevelGrowth = member?.nextLevelGrowth;
  const benefits = member?.benefits;

  const benefitItems = benefits
    ? Object.entries(BENEFIT_LABELS)
        .filter(([key]) => benefits[key as keyof BenefitVO] !== undefined)
        .map(([key, label]) => ({
          key,
          label,
          value: benefits[key as keyof BenefitVO],
        }))
    : [];

  const handleGoPackage = () => {
    Taro.navigateTo({ url: "/pages/personal/package/index" });
  };

  return (
    <PageLayout level="L2" title="我的会员">
      <View className="personal-member-page">
        <ScrollView scrollY className="member-scroll" enhanced showScrollbar={false}>
          {/* 会员头部卡片 */}
          <View className="member-header-card">
            <View className="member-avatar-section">
              <View className="avatar-wrapper">
                <Text className="avatar-icon">{levelCfg.icon}</Text>
              </View>
              <View className="member-info-section">
                <Text className="member-level-name">{levelCfg.name}</Text>
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
                <Text className="growth-value">{growthValue}</Text>
              </View>
              <View className="growth-bar-bg">
                <View className="growth-bar-fill" style={{ width: `${progressPercent}%` }} />
              </View>
              {nextLevelGrowth !== undefined && nextLevelGrowth > 0 && (
                <Text className="growth-tip">距下一级还需 {nextLevelGrowth} 成长值</Text>
              )}
            </View>
          </View>

          {/* 月度用量统计 */}
          <View className="usage-section">
            <View className="section-header">
              <Star size="16" color="#3b82f6" />
              <Text className="section-title">本月用量</Text>
            </View>
            <View className="usage-row">
              <View className="usage-item">
                <Text className="usage-num">{member?.monthlyDehazeUsed || 0}/{member?.monthlyDehazeQuota || 0}</Text>
                <Text className="usage-label">去雾处理</Text>
              </View>
              <View className="usage-item">
                <Text className="usage-num">{member?.monthlyEvaluateUsed || 0}/{member?.monthlyEvaluateQuota || 0}</Text>
                <Text className="usage-label">评估分析</Text>
              </View>
            </View>
          </View>

          {/* 会员权益 */}
          {benefitItems.length > 0 && (
            <View className="benefits-section">
              <View className="section-header">
                <Gift size="16" color="#3b82f6" />
                <Text className="section-title">会员权益</Text>
              </View>
              <View className="benefits-grid">
                {benefitItems.map((item) => (
                  <View key={item.key} className="benefit-card">
                    <Text className="benefit-value">{item.value}</Text>
                    <Text className="benefit-label">{item.label}</Text>
                  </View>
                ))}
              </View>
            </View>
          )}

          {/* 非会员引导 */}
          {(!member || member.levelCode === "level_0") && (
            <View className="non-vip-guide">
              <View className="guide-btn" onClick={handleGoPackage}>
                <Text className="guide-btn-text">开通 VIP</Text>
              </View>
            </View>
          )}

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
                      <Text className="log-desc">{log.reason || "获得成长值"}</Text>
                      <Text className="log-time">{new Date(log.createTime).toLocaleString()}</Text>
                    </View>
                    <Text className={`log-value ${log.changeValue > 0 ? "positive" : "negative"}`}>
                      {log.changeValue > 0 ? "+" : ""}{log.changeValue}
                    </Text>
                  </View>
                ))}
              </View>
            </View>
          )}

          <View className="member-footer">
            <Text className="footer-text">图像去雾系统 v1.0</Text>
          </View>
        </ScrollView>
      </View>
    </PageLayout>
  );
};

export default MemberPage;
