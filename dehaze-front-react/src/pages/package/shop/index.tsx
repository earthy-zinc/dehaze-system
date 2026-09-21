import {
  CouponAPI,
  MemberAPI,
  OrderAPI,
  PackageAPI,
  type MemberProfileVO,
  type PackageDetailVO,
  type PriceResult,
  type UserCouponVO,
} from "dehaze-sdk-js";
import {
  ArrowRightOutlined,
  CheckOutlined,
  CloseOutlined,
  GiftOutlined,
  ShoppingCartOutlined,
} from "@ant-design/icons";
import {
  Button,
  Empty,
  Modal,
  Select,
  Spin,
  Table,
  Tag,
  message,
} from "antd";
import type { TableColumnsType } from "antd";
import React, { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import "./index.scss";

const LEVEL_ORDER: Record<string, number> = {
  level_0: 0,
  level_1: 1,
  level_2: 2,
  level_3: 3,
};

const PERIOD_LABEL: Record<string, string> = {
  monthly: "月卡",
  quarterly: "季卡",
  yearly: "年卡",
};

const BENEFIT_LABELS: Record<string, string> = {
  monthlyDehazeQuota: "去雾配额",
  monthlyEvaluateQuota: "评估配额",
  historyRetention: "历史保留",
  batchLimit: "批量上限",
  priority: "优先级",
  advancedParams: "高级参数",
  hdExport: "高清导出",
  reportExport: "报告导出",
  batchDownload: "批量下载",
};

const BENEFIT_UNITS: Record<string, string> = {
  monthlyDehazeQuota: "次/月",
  monthlyEvaluateQuota: "次/月",
  historyRetention: "天",
  batchLimit: "张",
  priority: "级",
  advancedParams: "项",
  hdExport: "次",
  reportExport: "次",
  batchDownload: "次",
};

/** 后端金额单位为分，前端展示用元 */
const yuan = (cents?: number): string => ((cents ?? 0) / 100).toFixed(2);

const formatBenefitValue = (key: string, value: number): string => {
  const unit = BENEFIT_UNITS[key];
  if (key === "historyRetention") {
    return value === 0 ? "—" : `${value} ${unit}`;
  }
  if (key === "hdExport" || key === "reportExport" || key === "batchDownload") {
    if (value === 0) return "不支持";
    if (value === 1) return "支持";
  }
  if (key === "priority" || key === "advancedParams") {
    return value === 1 ? "支持" : "—";
  }
  if (unit) {
    return value === 0 ? "—" : `${value} ${unit}`;
  }
  return String(value);
};

interface ComparisonRow {
  label: string;
  benefitKey: string;
  [pkgKey: string]: string;
}

const PackageShop: React.FC = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [bannerVisible, setBannerVisible] = useState(true);
  const [packages, setPackages] = useState<PackageDetailVO[]>([]);
  const [profile, setProfile] = useState<MemberProfileVO | undefined>(
    undefined
  );

  /** 购买弹窗状态 */
  const [purchaseOpen, setPurchaseOpen] = useState(false);
  const [currentPackage, setCurrentPackage] = useState<PackageDetailVO | null>(
    null
  );
  const [priceResult, setPriceResult] = useState<PriceResult | null>(null);
  const [priceLoading, setPriceLoading] = useState(false);
  const [couponLoading, setCouponLoading] = useState(false);
  const [purchasing, setPurchasing] = useState(false);
  const [myCoupons, setMyCoupons] = useState<UserCouponVO[]>([]);
  const [selectedCouponId, setSelectedCouponId] = useState<
    number | undefined
  >(undefined);

  const currentLevelCode = profile?.levelCode ?? "level_0";

  const loadData = useCallback(() => {
    setLoading(true);
    Promise.all([
      PackageAPI.listOnSale(),
      MemberAPI.getProfile().catch(() => undefined),
    ])
      .then(([list, profileData]) => {
        setPackages(list || []);
        if (profileData) {
          setProfile(profileData);
        }
      })
      .finally(() => {
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const getButtonText = useCallback(
    (pkg: PackageDetailVO) => {
      if (pkg.packageType === "credit") {
        return "立即购买";
      }
      const current = LEVEL_ORDER[currentLevelCode] ?? 0;
      const target = LEVEL_ORDER[pkg.levelCode ?? ""] ?? 0;
      if (currentLevelCode === "level_0") {
        return "立即开通";
      }
      if (target === current) {
        return "续费";
      }
      if (target > current) {
        return "升级至此";
      }
      return "续费";
    },
    [currentLevelCode]
  );

  const recalcPrice = useCallback((pkgId: number, couponId?: number) => {
    setPriceLoading(true);
    return PackageAPI.calculatePrice(pkgId, couponId)
      .then((res) => {
        setPriceResult(res);
      })
      .catch(() => {
        setPriceResult(null);
      })
      .finally(() => {
        setPriceLoading(false);
      });
  }, []);

  /** 用户未使用优惠券中适用于当前套餐的（applicableScope 为空表示全部适用） */
  const applicableCoupons = useMemo(() => {
    if (!currentPackage) return [];
    return myCoupons.filter((coupon) => {
      const scope = coupon.applicableScope;
      if (!scope || scope.length === 0) return true;
      return (
        scope.includes(currentPackage.id) ||
        scope.includes(currentPackage.packageType)
      );
    });
  }, [myCoupons, currentPackage]);

  const couponOptionLabel = (coupon: UserCouponVO): string => {
    const threshold =
      coupon.threshold && coupon.threshold > 0
        ? `（满¥${yuan(coupon.threshold)}可用）`
        : "";
    return `${coupon.couponName}${threshold}`;
  };

  const handlePurchase = useCallback(
    (pkg: PackageDetailVO) => {
      setCurrentPackage(pkg);
      setSelectedCouponId(undefined);
      setPriceResult(null);
      setMyCoupons([]);
      setPurchaseOpen(true);
      setCouponLoading(true);
      CouponAPI.listMy(1)
        .then((coupons) => {
          setMyCoupons(coupons || []);
        })
        .catch(() => {
          setMyCoupons([]);
        })
        .finally(() => {
          setCouponLoading(false);
        });
      recalcPrice(pkg.id, undefined);
    },
    [recalcPrice]
  );

  const handleCouponChange = useCallback(
    (value?: number) => {
      setSelectedCouponId(value);
      if (currentPackage) {
        recalcPrice(currentPackage.id, value);
      }
    },
    [currentPackage, recalcPrice]
  );

  const confirmPurchase = useCallback(() => {
    if (!currentPackage || priceLoading || purchasing) return;
    setPurchasing(true);
    OrderAPI.create({
      packageId: currentPackage.id,
      couponId: selectedCouponId,
      payMethod: "balance",
    })
      .then((res) => {
        message.success("订单创建成功");
        setPurchaseOpen(false);
        navigate(`/order/detail?orderNo=${res.orderNo}`);
      })
      .catch((error) => {
        message.error(error?.message || "订单创建失败");
      })
      .finally(() => {
        setPurchasing(false);
      });
  }, [currentPackage, priceLoading, purchasing, selectedCouponId, navigate]);

  const comparisonPackages = useMemo(() => {
    const levelMap = new Map<string, PackageDetailVO>();
    packages.forEach((pkg) => {
      if (pkg.packageType === "vip" && pkg.levelCode && !levelMap.has(pkg.levelCode)) {
        levelMap.set(pkg.levelCode, pkg);
      }
    });
    const ordered: PackageDetailVO[] = [];
    ["level_1", "level_2", "level_3"].forEach((code) => {
      const pkg = levelMap.get(code);
      if (pkg) ordered.push(pkg);
    });
    return ordered;
  }, [packages]);

  const comparisonRows = useMemo<ComparisonRow[]>(() => {
    const allKeys = new Set<string>();
    comparisonPackages.forEach((pkg) => {
      Object.keys(pkg.benefits || {}).forEach((k) => allKeys.add(k));
    });
    return Array.from(allKeys).map((key) => {
      const row: ComparisonRow = {
        label: BENEFIT_LABELS[key] ?? key,
        benefitKey: key,
      };
      comparisonPackages.forEach((pkg) => {
        const val = pkg.benefits?.[key];
        row[`pkg_${pkg.id}`] =
          val !== undefined ? formatBenefitValue(key, Number(val)) : "—";
      });
      return row;
    });
  }, [comparisonPackages]);

  const comparisonColumns = useMemo<TableColumnsType<ComparisonRow>>(() => {
    const cols: TableColumnsType<ComparisonRow> = [
      {
        title: "权益项",
        dataIndex: "label",
        key: "label",
        width: 160,
        fixed: "left",
      },
    ];
    comparisonPackages.forEach((pkg) => {
      cols.push({
        title: pkg.levelName,
        key: `pkg_${pkg.id}`,
        align: "center",
        width: 140,
        render: (_: unknown, record: ComparisonRow) =>
          record[`pkg_${pkg.id}`] ?? "—",
      });
    });
    return cols;
  }, [comparisonPackages]);

  return (
    <div className="package-shop">
      <div className="shop-container">
        {bannerVisible && (
          <div className="promo-banner">
            <div className="banner-content">
              <GiftOutlined className="banner-icon" />
              <span className="banner-text">
                开通会员，解锁高清去雾、批量处理、专业评估等全部能力
              </span>
            </div>
            <CloseOutlined
              className="banner-close"
              onClick={() => setBannerVisible(false)}
            />
          </div>
        )}

        <div className="page-header">
          <h2 className="header-title">会员套餐</h2>
          <p className="header-subtitle">选择适合您的套餐，开启专业去雾体验</p>
        </div>

        <Spin spinning={loading}>
          <div className="package-cards">
            {packages.length > 0 ? (
              <div className="cards-row">
                {packages.map((pkg) => (
                  <div
                    key={pkg.id}
                    className={`package-card level-${pkg.levelCode ?? ""}`}
                  >
                    <div className="card-glow" />
                    <div className="card-inner">
                      <div className="card-header">
                        <div
                          className={`level-icon level-icon-${pkg.levelCode ?? ""}`}
                        >
                          <CheckOutlined />
                        </div>
                        <div className="level-info">
                          <div className="level-name">
                            {pkg.packageType === "credit"
                              ? "积分卡"
                              : pkg.levelName}
                          </div>
                          <div className="package-name">{pkg.name}</div>
                        </div>
                        {pkg.levelCode === currentLevelCode && (
                          <Tag color="success" className="current-badge">
                            当前
                          </Tag>
                        )}
                      </div>

                      {pkg.packageType === "credit" ? (
                        <div className="price-section">
                          <div className="credit-amount">
                            {pkg.creditAmount ?? 0}
                            <span className="credit-unit">积分</span>
                          </div>
                          <div className="credit-unit-price">
                            ¥{yuan(pkg.creditUnitPrice)}/积分
                          </div>
                          <div className="sale-price">
                            <span className="currency">¥</span>
                            <span className="price-num">
                              {yuan(pkg.salePrice)}
                            </span>
                            <span className="original-price">
                              原价 ¥{yuan(pkg.originalPrice)}
                            </span>
                          </div>
                        </div>
                      ) : (
                        <>
                          <div className="price-section">
                            <div className="sale-price">
                              <span className="currency">¥</span>
                              <span className="price-num">
                                {yuan(pkg.salePrice)}
                              </span>
                            </div>
                            <div className="original-price">
                              原价 ¥{yuan(pkg.originalPrice)}
                            </div>
                            <div className="daily-price">
                              ¥{yuan(pkg.dailyPrice)}/天
                              {pkg.period
                                ? ` · ${PERIOD_LABEL[pkg.period] ?? ""}`
                                : ""}
                            </div>
                          </div>

                          <div className="benefits-list">
                            {Object.entries(pkg.benefits || {}).map(
                              ([key, value]) => (
                                <div key={key} className="benefit-item">
                                  <CheckOutlined className="benefit-check" />
                                  <span className="benefit-label">
                                    {BENEFIT_LABELS[key] ?? key}
                                  </span>
                                  <span className="benefit-value">
                                    {formatBenefitValue(key, Number(value))}
                                  </span>
                                </div>
                              )
                            )}
                          </div>
                        </>
                      )}

                      {pkg.description && (
                        <div className="package-desc">{pkg.description}</div>
                      )}

                      <Button
                        type="primary"
                        className="action-btn"
                        icon={<ShoppingCartOutlined />}
                        onClick={() => handlePurchase(pkg)}
                      >
                        {getButtonText(pkg)}
                        <ArrowRightOutlined className="btn-icon" />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              !loading && (
                <Empty
                  description="暂无在售套餐"
                  image={Empty.PRESENTED_IMAGE_SIMPLE}
                />
              )
            )}
          </div>
        </Spin>

        {comparisonPackages.length > 0 && (
          <div className="comparison-section">
            <h3 className="section-title">权益对比</h3>
            <Table
              dataSource={comparisonRows}
              columns={comparisonColumns}
              rowKey={(record) => record.benefitKey}
              pagination={false}
              bordered
              size="middle"
              className="comparison-table"
              scroll={{ x: 600 }}
            />
          </div>
        )}
      </div>

      <Modal
        title="购买确认"
        open={purchaseOpen}
        width={480}
        maskClosable={false}
        onCancel={() => setPurchaseOpen(false)}
        footer={[
          <Button key="cancel" onClick={() => setPurchaseOpen(false)}>
            取消
          </Button>,
          <Button
            key="ok"
            type="primary"
            loading={purchasing}
            disabled={priceLoading}
            onClick={confirmPurchase}
          >
            确认购买
          </Button>,
        ]}
      >
        <Spin spinning={priceLoading}>
          <div className="purchase-detail">
            <div className="detail-row">
              <span className="detail-label">商品</span>
              <span className="detail-value">{currentPackage?.name}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">原价</span>
              <span className="detail-value">¥{yuan(priceResult?.originalPrice)}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">促销优惠</span>
              <span className="detail-value discount">
                -¥{yuan(priceResult?.discountAmount)}
              </span>
            </div>
            <div className="detail-row">
              <span className="detail-label">优惠券抵扣</span>
              <span className="detail-value discount">
                {priceResult?.couponAmount
                  ? `-¥${yuan(priceResult.couponAmount)}`
                  : "—"}
              </span>
            </div>
            <div className="detail-row coupon-select-row">
              <span className="detail-label">优惠券</span>
              <Select
                className="coupon-select"
                placeholder="不使用优惠券"
                allowClear
                value={selectedCouponId}
                loading={couponLoading}
                disabled={couponLoading}
                onChange={handleCouponChange}
                options={applicableCoupons.map((coupon) => ({
                  label: couponOptionLabel(coupon),
                  value: coupon.id,
                }))}
              />
            </div>
            <div className="detail-row total">
              <span className="detail-label">应付</span>
              <span className="detail-value payable">
                ¥{yuan(priceResult?.payableAmount)}
              </span>
            </div>
          </div>
        </Spin>
      </Modal>
    </div>
  );
};

export default PackageShop;
