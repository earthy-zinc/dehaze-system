import React, { useState, useEffect, useCallback } from "react";
import { View, Text, Input, Textarea } from "@tarojs/components";
import { Popup, Button, Switch, Cell } from "@taroify/core";
import { Arrow } from "@taroify/icons";
import type { Dataset, DatasetOption } from "dehaze-sdk-js";
import "./index.less";

export interface DatasetFormData {
  parentId: number;
  type?: string;
  name?: string;
  description?: string;
  status?: string;
}

interface DatasetFormDialogProps {
  visible: boolean;
  mode: "create" | "edit";
  dataset?: Dataset | null;
  options: DatasetOption[];
  defaultParentId?: number;
  onSubmit: (data: DatasetFormData) => Promise<boolean>;
  onClose: () => void;
}

// 数据集类型选项
const TYPE_OPTIONS = [
  { value: "training", label: "训练集" },
  { value: "test", label: "测试集" },
  { value: "user", label: "用户集" },
  { value: "result", label: "结果集" },
];

const DatasetFormDialog: React.FC<DatasetFormDialogProps> = ({
  visible,
  mode,
  dataset,
  options,
  defaultParentId = 0,
  onSubmit,
  onClose,
}) => {
  const [formData, setFormData] = useState<DatasetFormData>({
    parentId: 0,
    type: "user",
    name: "",
    description: "",
    status: "1",
  });
  const [showParentSelect, setShowParentSelect] = useState(false);
  const [selectedParent, setSelectedParent] = useState<DatasetOption | null>(
    null
  );
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  // 初始化/重置表单
  useEffect(() => {
    if (visible) {
      if (mode === "edit" && dataset) {
        setFormData({
          parentId: dataset.parentId ?? 0,
          type: dataset.type,
          name: dataset.name,
          description: dataset.description || "",
          status: String(dataset.status ?? 1),
        });
        setSelectedParent(
          options.find((o) => o.value === dataset.parentId) || null
        );
      } else {
        // 新增模式
        setFormData({
          parentId: defaultParentId,
          type: "user",
          name: "",
          description: "",
          status: "1",
        });
        setSelectedParent(
          options.find((o) => o.value === defaultParentId) || null
        );
      }
      setError("");
    }
  }, [visible, mode, dataset, defaultParentId, options]);

  const handleSubmit = useCallback(async () => {
    // 表单校验
    if (!formData.name?.trim()) {
      setError("请输入数据集名称");
      return;
    }
    setSubmitting(true);
    const success = await onSubmit(formData);
    setSubmitting(false);
    if (success) {
      onClose();
    }
  }, [formData, onSubmit, onClose]);

  return (
    <Popup
      open={visible}
      placement="bottom"
      style={{ height: "70%", borderRadius: "16px 16px 0 0" }}
      onClose={onClose}
    >
      <View className="dataset-form-dialog">
        {/* 头部 */}
        <View className="dialog-header">
          <Text className="dialog-title">
            {mode === "create" ? "新增数据集" : "修改数据集"}
          </Text>
          <View className="dialog-close" onClick={onClose}>
            <Text>✕</Text>
          </View>
        </View>

        {/* 表单内容 */}
        <View className="form-body">
          {/* 父级数据集 */}
          <Cell
            title="上级数据集"
            rightIcon={<Arrow size="16" color="#9ca3af" />}
            onClick={() => setShowParentSelect(true)}
          >
            <Text className="cell-value">
              {selectedParent ? selectedParent.label : "根数据集"}
            </Text>
          </Cell>

          {/* 数据集类型 */}
          <View className="form-item">
            <Text className="form-label">数据集类型</Text>
            <View className="type-selector">
              {TYPE_OPTIONS.map((opt) => (
                <View
                  key={opt.value}
                  className={`type-option ${formData.type === opt.value ? "active" : ""}`}
                  onClick={() =>
                    setFormData((prev) => ({ ...prev, type: opt.value }))
                  }
                >
                  <Text>{opt.label}</Text>
                </View>
              ))}
            </View>
          </View>

          {/* 名称 */}
          <View className="form-item">
            <Text className="form-label">
              数据集名称 <Text className="required">*</Text>
            </Text>
            <Input
              className="form-input"
              value={formData.name}
              placeholder="请输入数据集名称"
              onInput={(e) =>
                setFormData((prev) => ({ ...prev, name: e.detail.value }))
              }
            />
          </View>

          {/* 描述 */}
          <View className="form-item">
            <Text className="form-label">描述</Text>
            <Textarea
              className="form-textarea"
              value={formData.description}
              placeholder="请输入数据集描述（选填）"
              maxlength={200}
              onInput={(e) =>
                setFormData((prev) => ({
                  ...prev,
                  description: e.detail.value,
                }))
              }
            />
          </View>

          {/* 状态 */}
          <Cell title="启用状态">
            <Switch
              checked={formData.status === "1"}
              onChange={(checked) =>
                setFormData((prev) => ({
                  ...prev,
                  status: checked ? "1" : "0",
                }))
              }
            />
          </Cell>

          {error && (
            <View className="form-error">
              <Text>{error}</Text>
            </View>
          )}
        </View>

        {/* 底部操作 */}
        <View className="dialog-footer">
          <Button variant="outlined" onClick={onClose}>
            取消
          </Button>
          <Button color="primary" loading={submitting} onClick={handleSubmit}>
            确定
          </Button>
        </View>
      </View>

      {/* 父级选择弹窗 */}
      <Popup
        open={showParentSelect}
        placement="bottom"
        style={{ height: "50%", borderRadius: "16px 16px 0 0" }}
        onClose={() => setShowParentSelect(false)}
      >
        <View className="parent-select">
          <View className="select-header">
            <Text className="select-title">选择上级数据集</Text>
          </View>
          <View className="select-list">
            <View
              className={`select-item ${formData.parentId === 0 ? "active" : ""}`}
              onClick={() => {
                setFormData((prev) => ({ ...prev, parentId: 0 }));
                setSelectedParent(null);
                setShowParentSelect(false);
              }}
            >
              <Text>根数据集</Text>
            </View>
            {options.map((opt) => (
              <View
                key={opt.value}
                className={`select-item ${formData.parentId === opt.value ? "active" : ""}`}
                onClick={() => {
                  setFormData((prev) => ({
                    ...prev,
                    parentId: opt.value as number,
                  }));
                  setSelectedParent(opt);
                  setShowParentSelect(false);
                }}
              >
                <Text>{opt.label}</Text>
              </View>
            ))}
          </View>
        </View>
      </Popup>
    </Popup>
  );
};

export default DatasetFormDialog;
