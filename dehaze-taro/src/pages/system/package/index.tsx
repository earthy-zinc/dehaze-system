import React, { useState, useEffect, useCallback } from "react";
import { View, Text, ScrollView } from "@tarojs/components";
import Taro from "@tarojs/taro";
import { Tag, Loading, Empty, Popup, Input } from "@taroify/core";
import { PackageAPI } from "dehaze-sdk-js";
import type { PackagePageVO } from "dehaze-sdk-js";
import PageLayout from "@/layout";
import { usePermission } from "@/hooks/usePermission";
import { getErrorMessage } from "@/utils/error";
import "./index.less";

const PERIOD_LABELS: Record<string, string> = {
  monthly: "月卡",
  quarterly: "季卡",
  yearly: "年卡",
};

const LEVEL_LABELS: Record<string, string> = {
  level_1: "普通",
  level_2: "白银",
  level_3: "黄金",
};

const PackageManagePage: React.FC = () => {
  const { hasPermission } = usePermission();
  const canEdit = hasPermission("sys:package:edit");

  const [packages, setPackages] = useState<PackagePageVO[]>([]);
  const [loading, setLoading] = useState(false);
  const [total, setTotal] = useState(0);
  const [pageNum, setPageNum] = useState(1);
  const [keyword, setKeyword] = useState("");
  const [statusFilter, setStatusFilter] = useState<number | undefined>(
    undefined
  );

  const [editPopupVisible, setEditPopupVisible] = useState(false);
  const [editingPkg, setEditingPkg] = useState<PackagePageVO | null>(null);
  const [formName, setFormName] = useState("");
  const [formSalePrice, setFormSalePrice] = useState("");
  const [formOriginalPrice, setFormOriginalPrice] = useState("");
  const [formSort, setFormSort] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const fetchPackages = useCallback(
    async (page: number, kw: string, status?: number) => {
      setLoading(true);
      try {
        const params: any = { pageNum: page, pageSize: 15 };
        if (kw) params.name = kw;
        if (status !== undefined) params.status = status;
        const res = await PackageAPI.getPage(params);
        setPackages(res.list);
        setTotal(res.total);
        setPageNum(page);
      } catch (err: unknown) {
        Taro.showToast({
          title: getErrorMessage(err, "加载套餐列表失败"),
          icon: "none",
        });
      } finally {
        setLoading(false);
      }
    },
    []
  );

  useEffect(() => {
    fetchPackages(1, "", undefined);
  }, [fetchPackages]);

  const handleSearch = () => {
    fetchPackages(1, keyword, statusFilter);
  };

  const handleLoadMore = () => {
    if (packages.length < total) {
      fetchPackages(pageNum + 1, keyword, statusFilter);
    }
  };

  const handleToggleStatus = async (pkg: PackagePageVO) => {
    if (!canEdit) return;
    const newStatus = pkg.status === 1 ? 0 : 1;
    const label = newStatus === 0 ? "下架" : "上架";
    const res = await Taro.showModal({
      title: `确认${label}`,
      content: `确定要${label}套餐「${pkg.name}」吗？`,
    });
    if (!res.confirm) return;
    try {
      await PackageAPI.updateStatus(pkg.id, newStatus);
      Taro.showToast({ title: `${label}成功`, icon: "success" });
      fetchPackages(pageNum, keyword, statusFilter);
    } catch (err: unknown) {
      Taro.showToast({ title: getErrorMessage(err, "操作失败"), icon: "none" });
    }
  };

  const [formDesc, setFormDesc] = useState("");

  const openEdit = (pkg: PackagePageVO) => {
    setEditingPkg(pkg);
    setFormName(pkg.name);
    setFormSalePrice(String(pkg.salePrice));
    setFormOriginalPrice(String(pkg.originalPrice));
    setFormSort("");
    setFormDesc("");
    setEditPopupVisible(true);
  };

  const handleSave = async () => {
    if (!editingPkg) return;
    setSubmitting(true);
    try {
      await PackageAPI.update(editingPkg.id, {
        name: formName || editingPkg.name,
        levelCode: editingPkg.levelCode,
        period: editingPkg.period,
        periodDays: editingPkg.periodDays,
        originalPrice: Number(formOriginalPrice) || editingPkg.originalPrice,
        salePrice: Number(formSalePrice) || editingPkg.salePrice,
        description: formDesc || undefined,
        sort: Number(formSort) || 0,
      });
      Taro.showToast({ title: "保存成功", icon: "success" });
      setEditPopupVisible(false);
      fetchPackages(pageNum, keyword, statusFilter);
    } catch (err: unknown) {
      Taro.showToast({ title: getErrorMessage(err, "保存失败"), icon: "none" });
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <PageLayout level="L2" title="套餐管理">
      <View className="system-manage-page">
        {/* 搜索栏 */}
        <View className="search-bar">
          <Input
            className="search-input"
            placeholder="搜索套餐名称"
            value={keyword}
            onInput={(e) => setKeyword(e.detail.value)}
            onConfirm={handleSearch}
          />
          <View className="filter-row">
            <Tag
              color={statusFilter === undefined ? "primary" : "default"}
              size="small"
              onClick={() => {
                setStatusFilter(undefined);
                fetchPackages(1, keyword, undefined);
              }}
            >
              全部
            </Tag>
            <Tag
              color={statusFilter === 1 ? "primary" : "default"}
              size="small"
              onClick={() => {
                setStatusFilter(1);
                fetchPackages(1, keyword, 1);
              }}
            >
              在售
            </Tag>
            <Tag
              color={statusFilter === 0 ? "primary" : "default"}
              size="small"
              onClick={() => {
                setStatusFilter(0);
                fetchPackages(1, keyword, 0);
              }}
            >
              已下架
            </Tag>
          </View>
        </View>

        {/* 套餐列表 */}
        <ScrollView
          scrollY
          className="list-scroll"
          onScrollToLower={handleLoadMore}
        >
          {loading && packages.length === 0 ? (
            <View className="loading-wrapper">
              <Loading>加载中...</Loading>
            </View>
          ) : packages.length === 0 ? (
            <Empty>
              <Empty.Description>暂无套餐数据</Empty.Description>
            </Empty>
          ) : (
            packages.map((pkg) => (
              <View key={pkg.id} className="list-card">
                <View className="card-header">
                  <View className="card-title-row">
                    <Text className="card-name">{pkg.name}</Text>
                    <Tag
                      size="small"
                      color={pkg.status === 1 ? "success" : "default"}
                    >
                      {pkg.status === 1 ? "在售" : "已下架"}
                    </Tag>
                  </View>
                  <Text className="card-id">#{pkg.id}</Text>
                </View>
                <View className="card-meta">
                  <Text className="meta-item">
                    等级:{" "}
                    {pkg.levelName ||
                      LEVEL_LABELS[pkg.levelCode] ||
                      pkg.levelCode}
                  </Text>
                  <Text className="meta-item">
                    周期: {PERIOD_LABELS[pkg.period] || pkg.period} (
                    {pkg.periodDays}天)
                  </Text>
                </View>
                <View className="card-pricing">
                  <Text className="sale-price">¥{pkg.salePrice}</Text>
                  <Text className="original-price">¥{pkg.originalPrice}</Text>
                  <Text className="daily-price">日均 ¥{pkg.dailyPrice}</Text>
                </View>
                <View className="card-meta">
                  <Text className="meta-item">销量: {pkg.salesCount}</Text>
                  <Text className="meta-item">
                    创建: {new Date(pkg.createTime).toLocaleDateString("zh-CN")}
                  </Text>
                </View>
                {canEdit && (
                  <View className="card-actions">
                    <View className="action-btn" onClick={() => openEdit(pkg)}>
                      编辑
                    </View>
                    <View
                      className={`action-btn ${pkg.status === 1 ? "danger" : "primary"}`}
                      onClick={() => handleToggleStatus(pkg)}
                    >
                      {pkg.status === 1 ? "下架" : "上架"}
                    </View>
                  </View>
                )}
              </View>
            ))
          )}
          {packages.length > 0 && packages.length < total && (
            <View className="load-more" onClick={handleLoadMore}>
              <Text>加载更多</Text>
            </View>
          )}
        </ScrollView>

        {/* 编辑弹窗 */}
        <Popup
          open={editPopupVisible}
          placement="bottom"
          rounded
          onClose={() => setEditPopupVisible(false)}
        >
          <View className="popup-content">
            <View className="popup-header">
              <Text className="popup-title">编辑套餐</Text>
              <Text
                className="popup-close"
                onClick={() => setEditPopupVisible(false)}
              >
                ×
              </Text>
            </View>
            <View className="popup-body">
              <View className="form-item">
                <Text className="form-label">套餐名称</Text>
                <Input
                  className="form-input"
                  value={formName}
                  onInput={(e) => setFormName(e.detail.value)}
                />
              </View>
              <View className="form-item">
                <Text className="form-label">售价</Text>
                <Input
                  className="form-input"
                  type="number"
                  value={formSalePrice}
                  onInput={(e) => setFormSalePrice(e.detail.value)}
                />
              </View>
              <View className="form-item">
                <Text className="form-label">原价</Text>
                <Input
                  className="form-input"
                  type="number"
                  value={formOriginalPrice}
                  onInput={(e) => setFormOriginalPrice(e.detail.value)}
                />
              </View>
              <View className="form-item">
                <Text className="form-label">描述</Text>
                <Input
                  className="form-input"
                  value={formDesc}
                  onInput={(e) => setFormDesc(e.detail.value)}
                />
              </View>
              <View className="form-item">
                <Text className="form-label">排序</Text>
                <Input
                  className="form-input"
                  type="number"
                  value={formSort}
                  onInput={(e) => setFormSort(e.detail.value)}
                />
              </View>
              <View className="popup-confirm-btn" onClick={handleSave}>
                <Text>{submitting ? "保存中..." : "保存"}</Text>
              </View>
            </View>
          </View>
        </Popup>
      </View>
    </PageLayout>
  );
};

export default PackageManagePage;
