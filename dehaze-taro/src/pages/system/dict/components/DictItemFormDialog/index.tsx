import React from "react";
import { View, Text, Input, Textarea } from "@tarojs/components";
import { Popup, Button, Switch } from "@taroify/core";
import type { DictForm } from "dehaze-sdk-js";

interface DictItemFormDialogProps {
  open: boolean;
  editingId?: number;
  form: DictForm;
  submitting: boolean;
  onClose: () => void;
  onFormChange: (form: DictForm) => void;
  onSubmit: () => void;
}

const DictItemFormDialog: React.FC<DictItemFormDialogProps> = ({
  open,
  editingId,
  form,
  submitting,
  onClose,
  onFormChange,
  onSubmit,
}) => {
  return (
    <Popup
      open={open}
      onClose={onClose}
      placement="bottom"
      style={{ height: "70%" }}
    >
      <View className="form-popup">
        <View className="form-header">
          <Text className="form-title">
            {editingId ? "编辑字典数据" : "新增字典数据"}
          </Text>
        </View>
        <View className="form-body">
          <View className="form-item">
            <Text className="form-label">字典标签 *</Text>
            <Input
              className="form-input"
              placeholder="请输入字典标签"
              value={form.name || ""}
              onInput={(e) => onFormChange({ ...form, name: e.detail.value })}
            />
          </View>
          <View className="form-item">
            <Text className="form-label">字典键值 *</Text>
            <Input
              className="form-input"
              placeholder="请输入字典键值"
              value={form.value || ""}
              onInput={(e) => onFormChange({ ...form, value: e.detail.value })}
            />
          </View>
          <View className="form-item">
            <Text className="form-label">排序</Text>
            <Input
              className="form-input"
              type="number"
              placeholder="请输入排序值"
              value={String(form.sort || 1)}
              onInput={(e) =>
                onFormChange({
                  ...form,
                  sort: Number(e.detail.value) || 1,
                })
              }
            />
          </View>
          <View className="form-item">
            <Text className="form-label">是否默认</Text>
            <View className="form-switch">
              <Switch
                checked={form.defaulted === 1}
                onChange={(checked) =>
                  onFormChange({ ...form, defaulted: checked ? 1 : 0 })
                }
              />
              <Text>{form.defaulted === 1 ? "是" : "否"}</Text>
            </View>
          </View>
          <View className="form-item">
            <Text className="form-label">状态</Text>
            <View className="form-switch">
              <Switch
                checked={form.status === 1}
                onChange={(checked) =>
                  onFormChange({ ...form, status: checked ? 1 : 0 })
                }
              />
              <Text>{form.status === 1 ? "启用" : "禁用"}</Text>
            </View>
          </View>
          <View className="form-item">
            <Text className="form-label">备注</Text>
            <Textarea
              className="form-textarea"
              placeholder="请输入备注信息（最多200字符）"
              maxlength={200}
              value={form.remark || ""}
              onInput={(e) =>
                onFormChange({ ...form, remark: e.detail.value })
              }
            />
          </View>
        </View>
        <View className="form-footer">
          <Button onClick={onClose}>取消</Button>
          <Button color="primary" loading={submitting} onClick={onSubmit}>
            确定
          </Button>
        </View>
      </View>
    </Popup>
  );
};

export default DictItemFormDialog;
