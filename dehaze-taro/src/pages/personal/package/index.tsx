import React, { useState, useEffect, useCallback } from "react";
import { View, Text, ScrollView } from "@tarojs/components";
import Taro from "@tarojs/taro";
import { Gift } from "@taroify/icons";
import { PackageAPI } from "dehaze-sdk-js";
import type { PackageDetailVO } from "dehaze-sdk-js";
import PageLayout from "@/layout";
import "./index.less";

const PERIOD_LABELS: Record<string, string> = {
  monthly: "月",
  quarterly: "季",
  yearly: "年",
};

const PackagePage: React.FC = () => {
  const [packages, setPackages] = useState<PackageDetailVO[]>([]);
  const [loading, setLoading] = useState(false);

  const loadPackages = useCallback(async () => {
    setLoading(true);
    try {
      const list = await PackageAPI.listOnSale();
      setPackages(list || []);
    } catch {
      Taro.showToast({ title: "加载失败", icon: "none" });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadPackages();
  }, [loadPackages]);

  const handlePurchase = useCallback(async (pkg: PackageDetailVO) => {
    const confirmed = await Taro.showModal({
      title: "确认购买",
      content: `确定要购买「${pkg.name}」套餐吗？`,
      confirmText: "确认购买",
      cancelText: "取消",
      confirmColor: "#3b82f6",
    });
    if (!confirmed.confirm) return;
    Taro.showToast({ title: "支付功能开发中", icon: "none" });
  }, []);

  return (
    <PageLayout level="L2" title="我的套餐">
      <View className="personal-package-page">
        <ScrollView
          scrollY
          className="package-scroll"
          enhanced
          showScrollbar={false}
        >
          <View className="package-banner">
            <View className="banner-content">
              <Text className="banner-title">升级会员权益</Text>
              <Text className="banner-desc">
                解锁更多高级功能，享受极致去雾体验
              </Text>
            </View>
          </View>

          {loading ? (
            <View className="loading-wrapper">
              <Text>加载中...</Text>
            </View>
          ) : packages.length === 0 ? (
            <View className="empty-wrapper">
              <Text className="empty-icon">📦</Text>
              <Text className="empty-title">暂无可选套餐</Text>
            </View>
          ) : (
            <View className="package-list">
              {packages.map((pkg) => (
                <View key={pkg.id} className="package-card">
                  {pkg.name.includes("热门") && (
                    <View className="popular-badge">
                      <Gift size="12" color="#ffffff" />
                      <Text className="popular-text">热门推荐</Text>
                    </View>
                  )}

                  <View className="package-header">
                    <Text className="package-name">{pkg.name}</Text>
                    {pkg.description && (
                      <Text className="package-desc">{pkg.description}</Text>
                    )}
                  </View>

                  <View className="package-price">
                    <View className="price-row">
                      <Text className="price-currency">¥</Text>
                      <Text className="price-amount">{pkg.salePrice || 0}</Text>
                      <Text className="price-period">
                        /{PERIOD_LABELS[pkg.period || ""] || "月"}
                      </Text>
                    </View>
                    {pkg.originalPrice &&
                      pkg.originalPrice > (pkg.salePrice || 0) && (
                        <Text className="original-price">
                          原价 ¥{pkg.originalPrice}
                        </Text>
                      )}
                  </View>

                  {pkg.benefits && Object.keys(pkg.benefits).length > 0 && (
                    <View className="package-benefits">
                      {Object.entries(pkg.benefits).map(([key, val]) => (
                        <View key={key} className="benefit-tag">
                          <Text className="benefit-tag-text">
                            {key}: {val}
                          </Text>
                        </View>
                      ))}
                    </View>
                  )}

                  <View className="package-action">
                    <View
                      className="purchase-btn"
                      onClick={() => handlePurchase(pkg)}
                    >
                      <Text className="purchase-text">立即购买</Text>
                    </View>
                  </View>
                </View>
              ))}
            </View>
          )}
        </ScrollView>
      </View>
    </PageLayout>
  );
};

export default PackagePage;
