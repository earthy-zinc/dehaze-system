import React, { useState, useEffect, useCallback } from "react";
import { View, Text, ScrollView } from "@tarojs/components";
import Taro from "@tarojs/taro";
import { Navbar, Button, Popup, Input, Switch, Tag } from "@taroify/core";
import { ArrowLeft, Plus, Edit, Delete } from "@taroify/icons";
import { RecommendationAPI } from "dehaze-sdk-js";
import type { RecommendationRule } from "dehaze-sdk-js";
import PageLayout from "@/layout";
import EmptyState from "@/components/common/EmptyState";
import StatusTag from "@/components/common/StatusTag";
import "./index.less";

const RecommendRulesPage: React.FC = () => {
  const [rules, setRules] = useState<RecommendationRule[]>([]);
  const [loading, setLoading] = useState(false);

  // 弹窗状态
  const [editMode, setEditMode] = useState<"create" | "edit" | null>(null);
  const [editingRule, setEditingRule] = useState<RecommendationRule | null>(
    null
  );
  const [rulePopupVisible, setRulePopupVisible] = useState(false);

  // 表单数据
  const [formData, setFormData] = useState<{
    ruleName?: string;
    sceneType?: string;
    algorithmIds?: number;
    weight?: number;
    enabled?: boolean;
  }>({
    enabled: true,
    weight: 1,
  });

  // 加载规则列表
  const loadRules = useCallback(async () => {
    setLoading(true);
    try {
      const allRules = await RecommendationAPI.getRules();
      setRules(allRules || []);
    } catch {
      Taro.showToast({ title: "加载规则失败", icon: "none" });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadRules();
  }, [loadRules]);

  // 打开新增弹窗
  const handleCreate = () => {
    setEditMode("create");
    setEditingRule(null);
    setFormData({
      enabled: true,
      weight: 1,
    });
    setRulePopupVisible(true);
  };

  // 打开编辑弹窗
  const handleEdit = (rule: RecommendationRule) => {
    setEditMode("edit");
    setEditingRule(rule);
    setFormData({
      ruleName: rule.ruleName,
      sceneType: rule.sceneType,
      algorithmIds: rule.algorithmIds?.[0],
      weight: rule.weight ?? 1,
      enabled: rule.enabled ?? true,
    });
    setRulePopupVisible(true);
  };

  // 保存规则
  const handleSave = async () => {
    if (!formData.ruleName) {
      Taro.showToast({ title: "请输入规则名称", icon: "none" });
      return;
    }

    try {
      const payload = {
        ruleName: formData.ruleName,
        sceneType: formData.sceneType || undefined,
        algorithmIds: formData.algorithmIds
          ? [formData.algorithmIds]
          : undefined,
        weight: formData.weight || 1,
        enabled: formData.enabled ?? true,
      };

      if (editMode === "create") {
        await RecommendationAPI.updateRule(0, payload as any);
      } else {
        await RecommendationAPI.updateRule(
          editingRule?.id ?? 0,
          payload as any
        );
      }

      setRulePopupVisible(false);
      Taro.showToast({
        title: editMode === "create" ? "添加成功" : "保存成功",
        icon: "success",
      });
      loadRules();
    } catch {
      Taro.showToast({ title: "保存失败", icon: "none" });
    }
  };

  // 删除规则
  const handleDelete = useCallback(
    async (rule: RecommendationRule) => {
      const confirmed = await Taro.showModal({
        title: "确认删除",
        content: `确定要删除规则「${rule.ruleName}」吗？`,
        confirmColor: "#ef4444",
      });
      if (!confirmed.confirm) return;

      try {
        // TODO: 实现删除规则 API
        await RecommendationAPI.updateRule(rule.id!, {
          ...rule,
          enabled: false,
        });
        Taro.showToast({ title: "已删除", icon: "success" });
        loadRules();
      } catch {
        Taro.showToast({ title: "删除失败", icon: "none" });
      }
    },
    [loadRules]
  );

  // 场景类型标签
  const SCENE_LABELS: Record<string, string> = {
    outdoor: "户外",
    indoor: "室内",
    landscape: "风景",
    portrait: "人像",
    urban: "城市",
    nature: "自然",
    other: "其他",
  };

  return (
    <PageLayout level="L2" title="推荐规则">
      <View className="recommend-page">
        {/* 顶部操作栏 */}
        <View className="recommend-header">
          <Navbar.NavLeft>
            <ArrowLeft />
          </Navbar.NavLeft>
          <Navbar.Title>推荐规则</Navbar.Title>
          <Navbar.NavRight icon={<Plus size="20" />} onClick={handleCreate} />
        </View>

        {/* 规则统计 */}
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
              <Text className="loading-text">加载中...</Text>
            </View>
          ) : rules.length === 0 ? (
            <View className="empty-wrapper">
              <EmptyState
                type="search"
                title="暂无推荐规则"
                description="添加规则以自定义算法推荐逻辑"
              />
            </View>
          ) : (
            <>
              {rules.map((rule) => (
                <View key={rule.id} className="rule-card">
                  {/* 头部：规则名 + 状态 */}
                  <View className="rule-top">
                    <View className="rule-algo-name">
                      <Text className="algo-title">{rule.ruleName}</Text>
                    </View>
                    <StatusTag status={rule.enabled ? 1 : 0} size="small" />
                  </View>

                  {/* 条件信息 */}
                  <View className="rule-conditions">
                    {rule.sceneType && (
                      <Tag color="primary" size="small">
                        场景: {SCENE_LABELS[rule.sceneType] || rule.sceneType}
                      </Tag>
                    )}
                    {rule.weight && (
                      <Tag color="warning" size="small">
                        权重: {rule.weight}
                      </Tag>
                    )}
                  </View>

                  {/* 权重 */}
                  <View className="rule-meta">
                    <View className="meta-item">
                      <Text className="meta-label">权重:</Text>
                      <Text className="meta-value">{rule.weight ?? 1}</Text>
                    </View>
                  </View>

                  {/* 操作按钮 */}
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
                </View>
              ))}
            </>
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
              <Text className="popup-title">
                {editMode === "create" ? "添加推荐规则" : "编辑推荐规则"}
              </Text>
              <View
                className="popup-close"
                onClick={() => setRulePopupVisible(false)}
              >
                ×
              </View>
            </View>

            <ScrollView scrollY className="popup-scroll">
              {/* 规则名称 */}
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

              {/* 场景类型 */}
              <View className="form-item">
                <Text className="form-label">场景类型</Text>
                <Input
                  className="form-input"
                  placeholder="如: outdoor, indoor, landscape"
                  value={formData.sceneType || ""}
                  onInput={(e) =>
                    setFormData({ ...formData, sceneType: e.detail?.value })
                  }
                />
              </View>

              {/* 权重 */}
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

              {/* 启用开关 */}
              <View className="form-item switch-item">
                <Text className="form-label">启用规则</Text>
                <Switch
                  checked={formData.enabled ?? true}
                  onChange={(checked) =>
                    setFormData({
                      ...formData,
                      enabled: checked,
                    })
                  }
                />
              </View>
            </ScrollView>

            <View className="popup-actions">
              <Button
                variant="outlined"
                onClick={() => setRulePopupVisible(false)}
                className="popup-cancel-btn"
              >
                取消
              </Button>
              <Button
                variant="contained"
                onClick={handleSave}
                className="popup-confirm-btn"
              >
                保存
              </Button>
            </View>
          </View>
        </Popup>
      </View>
    </PageLayout>
  );
};

export default RecommendRulesPage;
