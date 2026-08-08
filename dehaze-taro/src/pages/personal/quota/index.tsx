import React, { useState, useEffect, useCallback } from "react";
import { View, Text, ScrollView } from "@tarojs/components";
import { ModelAPI } from "dehaze-sdk-js";
import type { PredictionQuota } from "dehaze-sdk-js";
import PageLayout from "@/layout";
import "./index.less";

const QuotaPage: React.FC = () => {
  const [quota, setQuota] = useState<PredictionQuota | null>(null);

  const loadQuota = useCallback(async () => {
    try {
      const data = await ModelAPI.getQuota();
      setQuota(data);
    } catch {
      // 静默
    }
  }, []);

  useEffect(() => {
    loadQuota();
  }, [loadQuota]);

  const remaining = quota?.remaining ?? 0;
  const total = quota?.total ?? 0;
  const used = quota?.used ?? Math.max(0, total - remaining);
  const percent = total > 0 ? Math.round((used / total) * 100) : 0;

  return (
    <PageLayout level="L2" title="我的额度">
      <View className="personal-quota-page">
        <ScrollView scrollY className="quota-scroll">
          {/* 额度总览 */}
          <View className="quota-overview-card">
            <View className="quota-amount">
              <Text className="quota-number">{remaining}</Text>
              <Text className="quota-unit">次</Text>
            </View>
            <Text className="quota-label">剩余处理次数</Text>
            <View className="quota-progress-bg">
              <View
                className="quota-progress-fill"
                style={{ width: `${percent}%` }}
              />
            </View>
            <View className="quota-stats">
              <Text className="quota-stat-item">已用 {used} 次</Text>
              <Text className="quota-stat-item">总量 {total} 次</Text>
            </View>
          </View>

          {/* 额度说明 */}
          <View className="quota-info-card">
            <Text className="quota-info-title">额度说明</Text>
            <View className="quota-info-item">
              <Text className="quota-info-label">•</Text>
              <Text className="quota-info-text">
                基础用户每月有固定免费处理次数
              </Text>
            </View>
            <View className="quota-info-item">
              <Text className="quota-info-label">•</Text>
              <Text className="quota-info-text">
                VIP 会员可获得更多处理次数
              </Text>
            </View>
            <View className="quota-info-item">
              <Text className="quota-info-label">•</Text>
              <Text className="quota-info-text">额度按自然月重置，不累计</Text>
            </View>
          </View>
        </ScrollView>
      </View>
    </PageLayout>
  );
};

export default QuotaPage;
