import React, { useState, useCallback } from "react";
import { View, Text, ScrollView } from "@tarojs/components";
import Taro from "@tarojs/taro";
import { Gift } from "@taroify/icons";
import { PackageAPI } from "dehaze-sdk-js";
import type { PackagePageVO as PackagePageVOType } from "dehaze-sdk-js";
import PageLayout from "@/layout";
import StatusTag from "@/components/common/StatusTag";
import "./index.less";

// 中文周期映射
const PERIOD_LABELS: Record<string, string> = {
  monthly: "月",
  quarterly: "季",
  yearly: "年",
};

// 用户端套餐展示类型（扩展 SDK 类型以包含页面所需字段）
type PackageDisplayVO = PackagePageVOType & {
  description?: string;
  price: number;
  popular: boolean;
  benefits: string[];
};

// 默认套餐数据（如果 API 返回为空）
const DEFAULT_PACKAGES: PackageDisplayVO[] = [
  {
    id: 1,
    name: "基础版",
    levelCode: "level_1",
    description: "适合个人用户日常使用",
    price: 0,
    originalPrice: 0,
    period: "monthly",
    periodDays: 30,
    dailyPrice: 0,
    salesCount: 0,
    benefits: ["每日 10 次去雾", "基础算法访问", "1GB 存储空间"],
    popular: false,
    status: 1,
    createTime: new Date().toISOString(),
  },
  {
    id: 2,
    name: "专业版",
    levelCode: "level_2",
    description: "适合专业摄影师和设计师",
    price: 29.9,
    originalPrice: 59.9,
    period: "monthly",
    periodDays: 30,
    dailyPrice: 0.99,
    salesCount: 0,
    benefits: [
      "每日 100 次去雾",
      "全部算法访问",
      "10GB 存储空间",
      "优先处理队列",
    ],
    popular: true,
    status: 1,
    createTime: new Date().toISOString(),
  },
  {
    id: 3,
    name: "企业版",
    levelCode: "level_3",
    description: "适合团队和企业用户",
    price: 99.9,
    originalPrice: 199.9,
    period: "monthly",
    periodDays: 30,
    dailyPrice: 3.3,
    salesCount: 0,
    benefits: [
      "无限次去雾",
      "全部算法 + 定制算法",
      "100GB 存储空间",
      "团队协作功能",
      "API 接口访问",
      "专属客服支持",
    ],
    popular: false,
    status: 1,
    createTime: new Date().toISOString(),
  },
] as PackageDisplayVO[];

const PackagePage: React.FC = () => {
  const [packages, setPackages] = useState<PackageDisplayVO[]>([]);

  // 加载套餐列表
  const loadPackages = useCallback(async () => {
    try {
      const res = await PackageAPI.getPage({
        pageNum: 1,
        pageSize: 20,
      });
      const list = (res.list as unknown as PackageDisplayVO[]) || [];
      setPackages(list.length > 0 ? list : DEFAULT_PACKAGES);
    } catch {
      setPackages(DEFAULT_PACKAGES);
    }
  }, []);

  React.useEffect(() => {
    loadPackages();
  }, [loadPackages]);

  // 购买确认
  const handlePurchase = useCallback(async (pkg: PackageDisplayVO) => {
    const confirmed = await Taro.showModal({
      title: "确认购买",
      content: `确定要购买「${pkg.name}」套餐吗？\n价格：¥${pkg.price}/${pkg.period}`,
      confirmText: "确认购买",
      cancelText: "取消",
      confirmColor: "#3b82f6",
    });
    if (!confirmed.confirm) return;

    Taro.showToast({
      title: "正在跳转支付...",
      icon: "loading",
      duration: 2000,
    });
    setTimeout(() => {
      Taro.showToast({ title: "支付功能开发中", icon: "none" });
    }, 2000);
  }, []);

  return (
    <PageLayout showTabbar currentRoute="/pages/package/index" title="会员套餐">
      <View className="package-page">
        {/* 顶部横幅 */}
        <View className="package-banner">
          <View className="banner-content">
            <Text className="banner-title">升级会员权益</Text>
            <Text className="banner-desc">
              解锁更多高级功能，享受极致去雾体验
            </Text>
          </View>
        </View>

        <ScrollView
          scrollY
          className="package-scroll"
          enhanced
          showScrollbar={false}
        >
          {/* 套餐列表 */}
          <View className="package-list">
            {packages.map((pkg) => (
              <View key={pkg.id} className="package-card">
                {/* 热门标签 */}
                {pkg.popular && (
                  <View className="popular-badge">
                    <Gift size="12" color="#ffffff" />
                    <Text className="popular-text">热门推荐</Text>
                  </View>
                )}

                {/* 套餐信息 */}
                <View className="package-header">
                  <View className="package-name-row">
                    <Text className="package-name">{pkg.name}</Text>
                    <StatusTag status={pkg.status} size="small" />
                  </View>
                  <Text className="package-desc">{pkg.description}</Text>
                </View>

                {/* 价格 */}
                <View className="package-price">
                  <View className="price-row">
                    <Text className="price-currency">¥</Text>
                    <Text className="price-amount">{pkg.price}</Text>
                    <Text className="price-period">
                      /{PERIOD_LABELS[pkg.period] || pkg.period}
                    </Text>
                  </View>
                  {pkg.originalPrice > pkg.price && (
                    <Text className="original-price">
                      原价 ¥{pkg.originalPrice}
                    </Text>
                  )}
                </View>

                {/* 权益列表 */}
                <View className="package-benefits">
                  {(pkg.benefits || []).map((benefit: string, idx: number) => (
                    <View key={idx} className="benefit-item">
                      <Text className="benefit-check">✓</Text>
                      <Text className="benefit-text">{benefit}</Text>
                    </View>
                  ))}
                </View>

                {/* 购买按钮 */}
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
        </ScrollView>
      </View>
    </PageLayout>
  );
};

export default PackagePage;
