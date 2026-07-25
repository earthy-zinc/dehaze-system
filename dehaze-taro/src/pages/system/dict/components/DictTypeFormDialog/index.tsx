import React from "react";
import { View, Text, Input, Textarea } from "@tarojs/components";
import { Popup, Button, Switch } from "@taroify/core";
import type { DictTypeForm } from "dehaze-sdk-js";

interface DictTypeFormDialogProps {
  open: boolean;
  editingId?: number;
  form: DictTypeForm;
  submitting: boolean;
  onClose: () => void;
  onFormChange: (form: DictTypeForm) => void;
  onSubmit: () => void;
}

const DictTypeFormDialog: React.FC<DictTypeFormDialogProps> = ({
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
      style={{ height: "60%" }}
    >
      <View className="form-popup">
        <View className="form-header">
          <Text className="form-title">
            {editingId ? "编辑字典类型" : "新增字典类型"}
          </Text>
        </View>
        <View className="form-body">
          <View className="form-item">
            <Text className="form-label">字典名称 *</Text>
            <Input
              className="form-input"
              placeholder="请输入字典名称"
              value={form.name || ""}
              onInput={(e) => onFormChange({ ...form, name: e.detail.value })}
            />
          </View>
          <View className="form-item">
            <Text className="form-label">字典编码 *</Text>
            <Input
              className="form-input"
              placeholder="请输入字典编码"
              value={form.code || ""}
              disabled={!!editingId}
              onInput={(e) => onFormChange({ ...form, code: e.detail.value })}
            />
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
              onInput={(e) => onFormChange({ ...form, remark: e.detail.value })}
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

export default DictTypeFormDialog;
