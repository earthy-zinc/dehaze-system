import React from "react";
import { View, Text, Input, ScrollView } from "@tarojs/components";
import { Popup, Button, Switch } from "@taroify/core";
import { MenuTypeEnum } from "dehaze-sdk-js";
import type { MenuForm } from "dehaze-sdk-js";
import { MENU_TYPE_OPTIONS } from "../../constants";

interface MenuFormDialogProps {
  open: boolean;
  editingId?: string;
  form: MenuForm;
  submitting: boolean;
  menuOptions: { label: string; value: string | number }[];
  onClose: () => void;
  onFieldChange: (field: keyof MenuForm, value: string | number) => void;
  onTypeChange: (type: MenuTypeEnum) => void;
  onSubmit: () => void;
}

const MenuFormDialog: React.FC<MenuFormDialogProps> = ({
  open,
  editingId,
  form,
  submitting,
  menuOptions,
  onClose,
  onFieldChange,
  onTypeChange,
  onSubmit,
}) => {
  // 根据菜单类型动态显示字段
  const showPath =
    form.type === MenuTypeEnum.MENU || form.type === MenuTypeEnum.EXTLINK;
  const showComponent = form.type === MenuTypeEnum.MENU;
  const showPerm = form.type === MenuTypeEnum.BUTTON;
  const showIcon =
    form.type === MenuTypeEnum.CATALOG || form.type === MenuTypeEnum.MENU;
  const showRedirect = form.type === MenuTypeEnum.CATALOG;

  return (
    <Popup
      open={open}
      onClose={onClose}
      placement="bottom"
      style={{ height: "85%" }}
    >
      <View className="form-popup">
        <View className="form-header">
          <Text className="form-title">
            {editingId ? "编辑菜单" : "新增菜单"}
          </Text>
        </View>
        <ScrollView scrollY className="form-body">
          {/* 菜单类型 */}
          <View className="form-item">
            <Text className="form-label">菜单类型 *</Text>
            <View className="type-options">
              {MENU_TYPE_OPTIONS.map((option) => (
                <View
                  key={option.value}
                  className={`type-option ${form.type === option.value ? "active" : ""}`}
                  onClick={() => onTypeChange(option.value)}
                >
                  <Text>{option.label}</Text>
                </View>
              ))}
            </View>
          </View>

          {/* 上级菜单 */}
          <View className="form-item">
            <Text className="form-label">上级菜单</Text>
            <View className="parent-menu-display">
              {form.parentId === 0
                ? "顶级菜单"
                : menuOptions.find(
                    (opt) => Number(opt.value) === form.parentId
                  )?.label || "未知菜单"}
            </View>
          </View>

          {/* 菜单名称 */}
          <View className="form-item">
            <Text className="form-label">菜单名称 *</Text>
            <Input
              className="form-input"
              placeholder="请输入菜单名称"
              value={form.name || ""}
              onInput={(e) => onFieldChange("name", e.detail.value)}
            />
          </View>

          {/* 路由地址（菜单/外链） */}
          {showPath && (
            <View className="form-item">
              <Text className="form-label">
                {form.type === MenuTypeEnum.EXTLINK
                  ? "外链地址 *"
                  : "路由地址 *"}
              </Text>
              <Input
                className="form-input"
                placeholder={
                  form.type === MenuTypeEnum.EXTLINK
                    ? "请输入外链地址（https://）"
                    : "请输入路由地址（/开头）"
                }
                value={form.path || ""}
                onInput={(e) => onFieldChange("path", e.detail.value)}
              />
            </View>
          )}

          {/* 组件路径（菜单） */}
          {showComponent && (
            <View className="form-item">
              <Text className="form-label">组件路径 *</Text>
              <Input
                className="form-input"
                placeholder="请输入组件路径"
                value={form.component || ""}
                onInput={(e) => onFieldChange("component", e.detail.value)}
              />
            </View>
          )}

          {/* 权限标识（按钮） */}
          {showPerm && (
            <View className="form-item">
              <Text className="form-label">权限标识 *</Text>
              <Input
                className="form-input"
                placeholder="格式：模块:功能:操作（如 sys:menu:add）"
                value={form.perm || ""}
                onInput={(e) => onFieldChange("perm", e.detail.value)}
              />
            </View>
          )}

          {/* 图标（目录/菜单） */}
          {showIcon && (
            <View className="form-item">
              <Text className="form-label">图标</Text>
              <Input
                className="form-input"
                placeholder="请输入图标名称"
                value={form.icon || ""}
                onInput={(e) => onFieldChange("icon", e.detail.value)}
              />
            </View>
          )}

          {/* 路由重定向（目录） */}
          {showRedirect && (
            <View className="form-item">
              <Text className="form-label">路由重定向</Text>
              <Input
                className="form-input"
                placeholder="请输入路由重定向地址"
                value={form.redirect || ""}
                onInput={(e) => onFieldChange("redirect", e.detail.value)}
              />
            </View>
          )}

          {/* 排序 */}
          <View className="form-item">
            <Text className="form-label">排序</Text>
            <Input
              className="form-input"
              type="number"
              placeholder="请输入排序值"
              value={String(form.sort ?? 1)}
              onInput={(e) => onFieldChange("sort", Number(e.detail.value) || 1)}
            />
          </View>

          {/* 显示状态 */}
          <View className="form-item">
            <Text className="form-label">显示状态</Text>
            <View className="form-switch">
              <Switch
                checked={form.visible === 1}
                onChange={(checked) =>
                  onFieldChange("visible", checked ? 1 : 0)
                }
              />
              <Text>{form.visible === 1 ? "显示" : "隐藏"}</Text>
            </View>
          </View>
        </ScrollView>
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

export default MenuFormDialog;
