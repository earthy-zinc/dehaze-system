import {
  MemberAPI,
  OrderAPI,
  PackageAPI,
  type MemberProfileVO,
  type PackageDetailVO,
} from "dehaze-sdk-js";
import {
  ArrowRightOutlined,
  CheckOutlined,
  CloseOutlined,
  GiftOutlined,
  ShoppingCartOutlined,
} from "@ant-design/icons";
import { Button, Empty, Modal, Spin, Table, Tag, message } from "antd";
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
  const [purchasingId, setPurchasingId] = useState<number>(0);
  const [bannerVisible, setBannerVisible] = useState(true);
  const [packages, setPackages] = useState<PackageDetailVO[]>([]);
  const [profile, setProfile] = useState<MemberProfileVO | undefined>(
    undefined
  );

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
    (pkgLevel: string) => {
      const current = LEVEL_ORDER[currentLevelCode] ?? 0;
      const target = LEVEL_ORDER[pkgLevel] ?? 0;
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

  const getButtonType = useCallback(
    (pkgLevel: string): "default" | "primary" => {
      if (pkgLevel === "level_3") return "primary";
      if (pkgLevel === "level_2") return "primary";
      return "primary";
    },
    []
  );

  const handlePurchase = useCallback(
    (pkg: PackageDetailVO) => {
      Modal.confirm({
        title: "开通确认",
        content: `确认开通「${pkg.name}」，将使用余额支付 ¥${pkg.salePrice.toFixed(2)}？`,
        okText: "确认开通",
        cancelText: "取消",
        onOk: () => {
          setPurchasingId(pkg.id);
          return OrderAPI.create({
            packageId: pkg.id,
            payMethod: "balance",
          })
            .then((res) => {
              message.success("订单创建成功");
              navigate(`/order/detail?orderNo=${res.orderNo}`);
            })
            .catch((error) => {
              message.error(error?.message || "订单创建失败");
              return Promise.reject(error);
            })
            .finally(() => {
              setPurchasingId(0);
            });
        },
      });
    },
    [navigate]
  );

  const comparisonPackages = useMemo(() => {
    const levelMap = new Map<string, PackageDetailVO>();
    packages.forEach((pkg) => {
      if (!levelMap.has(pkg.levelCode)) {
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
                    className={`package-card level-${pkg.levelCode}`}
                  >
                    <div className="card-glow" />
                    <div className="card-inner">
                      <div className="card-header">
                        <div
                          className={`level-icon level-icon-${pkg.levelCode}`}
                        >
                          <CheckOutlined />
                        </div>
                        <div className="level-info">
                          <div className="level-name">{pkg.levelName}</div>
                          <div className="package-name">{pkg.name}</div>
                        </div>
                        {pkg.levelCode === currentLevelCode && (
                          <Tag color="success" className="current-badge">
                            当前
                          </Tag>
                        )}
                      </div>

                      <div className="price-section">
                        <div className="sale-price">
                          <span className="currency">¥</span>
                          <span className="price-num">
                            {pkg.salePrice.toFixed(2)}
                          </span>
                        </div>
                        <div className="original-price">
                          原价 ¥{pkg.originalPrice.toFixed(2)}
                        </div>
                        <div className="daily-price">
                          ¥{pkg.dailyPrice.toFixed(2)}/天 ·{" "}
                          {PERIOD_LABEL[pkg.period] ?? pkg.period}
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

                      {pkg.description && (
                        <div className="package-desc">{pkg.description}</div>
                      )}

                      <Button
                        type={getButtonType(pkg.levelCode)}
                        className="action-btn"
                        loading={purchasingId === pkg.id}
                        icon={<ShoppingCartOutlined />}
                        onClick={() => handlePurchase(pkg)}
                      >
                        {getButtonText(pkg.levelCode)}
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
    </div>
  );
};

export default PackageShop;
