import React, { useState, useEffect, useCallback } from "react";
import { View, Text, ScrollView } from "@tarojs/components";
import { AiBillingAPI } from "dehaze-sdk-js";
import type { BalanceVO } from "dehaze-sdk-js";
import PageLayout from "@/layout";
import "./index.less";

const QuotaPage: React.FC = () => {
  const [balance, setBalance] = useState<BalanceVO | null>(null);

  const loadBalance = useCallback(async () => {
    try {
      setBalance(await AiBillingAPI.getBalance());
    } catch {
      // 静默
    }
  }, []);

  useEffect(() => {
    loadBalance();
  }, [loadBalance]);

  const credits = balance?.creditsBalance ?? 0;
  const monthlyUsed = balance?.monthlyUsed ?? 0;
  const monthlyLimit = balance?.monthlyLimit ?? 0;
  const percent =
    monthlyLimit > 0 ? Math.round((monthlyUsed / monthlyLimit) * 100) : 0;

  return (
    <PageLayout level="L2" title="积分余额">
      <View className="personal-quota-page">
        <ScrollView scrollY className="quota-scroll">
          {/* 积分总览 */}
          <View className="quota-overview-card">
            <View className="quota-amount">
              <Text className="quota-number">{credits}</Text>
              <Text className="quota-unit">积分</Text>
            </View>
            <Text className="quota-label">剩余积分</Text>
            <View className="quota-progress-bg">
              <View
                className="quota-progress-fill"
                style={{ width: `${percent}%` }}
              />
            </View>
            <View className="quota-stats">
              <Text className="quota-stat-item">本月已用 {monthlyUsed}</Text>
              <Text className="quota-stat-item">本月限额 {monthlyLimit}</Text>
            </View>
          </View>

          {/* 积分说明 */}
          <View className="quota-info-card">
            <Text className="quota-info-title">积分说明</Text>
            <View className="quota-info-item">
              <Text className="quota-info-label">•</Text>
              <Text className="quota-info-text">
                图像去雾按张扣减积分，处理失败或取消不计费
              </Text>
            </View>
            <View className="quota-info-item">
              <Text className="quota-info-label">•</Text>
              <Text className="quota-info-text">
                VIP 会员每月获赠积分，等级越高赠送越多
              </Text>
            </View>
            <View className="quota-info-item">
              <Text className="quota-info-label">•</Text>
              <Text className="quota-info-text">
                积分余额不足时无法提交新的处理任务
              </Text>
            </View>
          </View>
        </ScrollView>
      </View>
    </PageLayout>
  );
};

export default QuotaPage;
