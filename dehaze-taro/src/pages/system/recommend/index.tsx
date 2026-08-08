import React, { useState, useEffect, useCallback } from "react";
import { View, Text, ScrollView } from "@tarojs/components";
import Taro from "@tarojs/taro";
import { Popup, Input, Switch, Tag, Loading, Empty } from "@taroify/core";
import { Plus, Edit, Delete } from "@taroify/icons";
import { RecommendationAPI } from "dehaze-sdk-js";
import type { RecommendationRule } from "dehaze-sdk-js";
import PageLayout from "@/layout";
import { usePermission } from "@/hooks/usePermission";
import { getErrorMessage } from "@/utils/error";
import "./index.less";

const SCENE_LABELS: Record<string, string> = {
  outdoor: "户外",
  indoor: "室内",
  landscape: "风景",
  portrait: "人像",
  urban: "城市",
  nature: "自然",
  other: "其他",
};

const RecommendManagePage: React.FC = () => {
  const { hasPermission } = usePermission();
  const canView = hasPermission("sys:recommendation:*");
  const canEdit = hasPermission("sys:recommendation:rule:edit");

  const [rules, setRules] = useState<RecommendationRule[]>([]);
  const [loading, setLoading] = useState(false);

  const [editingRule, setEditingRule] = useState<RecommendationRule | null>(
    null
  );
  const [rulePopupVisible, setRulePopupVisible] = useState(false);

  const [formData, setFormData] = useState<{
    ruleName?: string;
    sceneType?: string;
    algorithmIds?: string;
    weight?: number;
    enabled?: boolean;
  }>({ enabled: true, weight: 1 });

  const loadRules = useCallback(async () => {
    setLoading(true);
    try {
      const allRules = await RecommendationAPI.getRules();
      setRules(allRules || []);
    } catch (err: unknown) {
      Taro.showToast({
        title: getErrorMessage(err, "加载规则失败"),
        icon: "none",
      });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadRules();
  }, [loadRules]);

  const handleEdit = (rule: RecommendationRule) => {
    setEditingRule(rule);
    setFormData({
      ruleName: rule.ruleName,
      sceneType: rule.sceneType,
      algorithmIds: rule.algorithmIds?.join(",") || "",
      weight: rule.weight ?? 1,
      enabled: rule.enabled ?? true,
    });
    setRulePopupVisible(true);
  };

  const handleSave = async () => {
    if (!formData.ruleName) {
      Taro.showToast({ title: "请输入规则名称", icon: "none" });
      return;
    }
    if (!editingRule?.id) {
      Taro.showToast({ title: "规则ID无效", icon: "none" });
      return;
    }
    try {
      const algoIds = formData.algorithmIds
        ? formData.algorithmIds
            .split(",")
            .map((s) => Number(s.trim()))
            .filter((n) => !isNaN(n))
        : [];
      await RecommendationAPI.updateRule(editingRule.id, {
        id: editingRule.id,
        ruleName: formData.ruleName,
        sceneType: formData.sceneType || "",
        algorithmIds: algoIds,
        weight: formData.weight || 1,
        enabled: formData.enabled ?? true,
      });
      setRulePopupVisible(false);
      Taro.showToast({ title: "保存成功", icon: "success" });
      loadRules();
    } catch (err: unknown) {
      Taro.showToast({ title: getErrorMessage(err, "保存失败"), icon: "none" });
    }
  };

  const handleDelete = useCallback(
    async (rule: RecommendationRule) => {
      const res = await Taro.showModal({
        title: "确认删除",
        content: `确定要删除规则「${rule.ruleName}」吗？`,
        confirmColor: "#ef4444",
      });
      if (!res.confirm) return;
      try {
        await RecommendationAPI.updateRule(rule.id!, {
          ...rule,
          enabled: false,
        });
        Taro.showToast({ title: "已删除", icon: "success" });
        loadRules();
      } catch (err: unknown) {
        Taro.showToast({
          title: getErrorMessage(err, "删除失败"),
          icon: "none",
        });
      }
    },
    [loadRules]
  );

  if (!canView) {
    return (
      <PageLayout level="L2" title="推荐管理">
        <View className="no-permission">
          <Text>无权限访问</Text>
        </View>
      </PageLayout>
    );
  }

  return (
    <PageLayout level="L2" title="推荐管理">
      <View className="recommend-page">
        {/* 统计栏 */}
        <View className="stats-bar">
          <View className="stat-item">
            <Text className="stat-value">{rules.length}</Text>
            <Text className="stat-label">总规则数</Text>
          </View>
          <View className="stat-item">
            <Text className="stat-value">
              {rules.filter((r) => r.enabled).length}
            </Text>
            <Text className="stat-label">生效中</Text>
          </View>
          {canEdit && (
            <View className="stat-item create-stat">
              <Plus size="20" color="#3b82f6" />
              <Text className="stat-label">规则列表</Text>
            </View>
          )}
        </View>

        {/* 规则列表 */}
        <ScrollView
          scrollY
          className="rule-list"
          enhanced
          showScrollbar={false}
        >
          {loading && rules.length === 0 ? (
            <View className="loading-wrapper">
              <Loading>加载中...</Loading>
            </View>
          ) : rules.length === 0 ? (
            <View className="empty-wrapper">
              <Empty>
                <Empty.Description>暂无推荐规则</Empty.Description>
              </Empty>
            </View>
          ) : (
            rules.map((rule) => (
              <View key={rule.id} className="rule-card">
                <View className="rule-top">
                  <View className="rule-algo-name">
                    <Text className="algo-title">{rule.ruleName}</Text>
                  </View>
                  <Tag
                    size="small"
                    color={rule.enabled ? "success" : "default"}
                  >
                    {rule.enabled ? "生效中" : "已禁用"}
                  </Tag>
                </View>
                <View className="rule-conditions">
                  {rule.sceneType && (
                    <Tag color="primary" size="small">
                      场景: {SCENE_LABELS[rule.sceneType] || rule.sceneType}
                    </Tag>
                  )}
                  {rule.weight !== undefined && (
                    <Tag color="warning" size="small">
                      权重: {rule.weight}
                    </Tag>
                  )}
                </View>
                {canEdit && (
                  <View className="rule-actions">
                    <View
                      className="action-edit"
                      onClick={() => handleEdit(rule)}
                    >
                      <Edit size="14" color="#3b82f6" />
                      <Text className="action-text">编辑</Text>
                    </View>
                    <View
                      className="action-delete"
                      onClick={() => handleDelete(rule)}
                    >
                      <Delete size="14" color="#ef4444" />
                      <Text className="action-text">删除</Text>
                    </View>
                  </View>
                )}
              </View>
            ))
          )}
        </ScrollView>

        {/* 添加/编辑弹窗 */}
        <Popup
          open={rulePopupVisible}
          placement="bottom"
          rounded
          onClose={() => setRulePopupVisible(false)}
        >
          <View className="rule-popup">
            <View className="popup-title-bar">
              <Text className="popup-title">编辑推荐规则</Text>
              <Text
                className="popup-close"
                onClick={() => setRulePopupVisible(false)}
              >
                ×
              </Text>
            </View>
            <ScrollView scrollY className="popup-scroll">
              <View className="form-item">
                <Text className="form-label">规则名称 *</Text>
                <Input
                  className="form-input"
                  placeholder="请输入规则名称"
                  value={formData.ruleName || ""}
                  onInput={(e) =>
                    setFormData({ ...formData, ruleName: e.detail?.value })
                  }
                />
              </View>
              <View className="form-item">
                <Text className="form-label">场景类型</Text>
                <Input
                  className="form-input"
                  placeholder="如: urban, landscape, building"
                  value={formData.sceneType || ""}
                  onInput={(e) =>
                    setFormData({ ...formData, sceneType: e.detail?.value })
                  }
                />
              </View>
              <View className="form-item">
                <Text className="form-label">关联算法ID（逗号分隔）</Text>
                <Input
                  className="form-input"
                  placeholder="如: 1,2,3"
                  value={formData.algorithmIds || ""}
                  onInput={(e) =>
                    setFormData({
                      ...formData,
                      algorithmIds: e.detail?.value,
                    })
                  }
                />
              </View>
              <View className="form-item">
                <Text className="form-label">匹配权重 ({formData.weight})</Text>
                <Input
                  className="form-input"
                  type="number"
                  value={String(formData.weight || 1)}
                  onInput={(e) =>
                    setFormData({
                      ...formData,
                      weight: Number(e.detail?.value) || 1,
                    })
                  }
                />
              </View>
              <View className="form-item switch-item">
                <Text className="form-label">启用规则</Text>
                <Switch
                  checked={formData.enabled ?? true}
                  onChange={(checked) =>
                    setFormData({ ...formData, enabled: checked })
                  }
                />
              </View>
            </ScrollView>
            <View className="popup-actions">
              <View
                className="popup-cancel-btn"
                onClick={() => setRulePopupVisible(false)}
              >
                <Text>取消</Text>
              </View>
              <View className="popup-confirm-btn" onClick={handleSave}>
                <Text>保存</Text>
              </View>
            </View>
          </View>
        </Popup>
      </View>
    </PageLayout>
  );
};

export default RecommendManagePage;
