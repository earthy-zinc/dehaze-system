import React, { useState } from "react";
import { View } from "@tarojs/components";
import Taro, { useRouter, useLoad } from "@tarojs/taro";
import {
  Navbar,
  Form,
  Input,
  Radio,
  Stepper,
  Button,
  Loading,
  Cell,
  Field,
} from "@taroify/core";
import { ArrowLeft } from "@taroify/icons";
import { useRoleManagement } from "@/hooks/useRoleManagement";
import type { RoleForm } from "dehaze-sdk-js";
import "./detail.scss";

const RoleDetailPage: React.FC = () => {
  const router = useRouter();
  const { id } = router.params;
  const isEdit = !!id;

  const { createRole, updateRole } = useRoleManagement();

  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [formData, setFormData] = useState<RoleForm>({
    name: "",
    code: "",
    sort: 0,
    status: 1,
    dataScope: 1,
  });

  // 页面加载时初始化数据
  useLoad(async () => {
    // 如果是编辑模式，获取角色详情
    if (isEdit) {
      await loadRoleData();
    }
  });

  // 加载角色数据
  const loadRoleData = async () => {
    if (!id) return;

    try {
      setLoading(true);
      const { RoleAPI } = await import("dehaze-sdk-js");
      const roleData = await RoleAPI.getFormData(Number(id));
      setFormData({
        name: roleData.name || "",
        code: roleData.code || "",
        sort: roleData.sort || 0,
        status: roleData.status ?? 1,
        dataScope: roleData.dataScope || 1,
      });
    } catch (error) {
      Taro.showToast({ title: "获取角色信息失败", icon: "none" });
      console.error("获取角色信息失败:", error);
    } finally {
      setLoading(false);
    }
  };

  // 表单字段更新
  const handleFieldChange = (field: keyof RoleForm, value: any) => {
    setFormData((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  // 表单验证
  const validateForm = () => {
    if (!formData.name.trim()) {
      Taro.showToast({ title: "请输入角色名称", icon: "none" });
      return false;
    }

    if (!formData.code.trim()) {
      Taro.showToast({ title: "请输入角色编码", icon: "none" });
      return false;
    }

    // 验证编码格式（只允许字母、数字、下划线）
    if (!/^[a-zA-Z0-9_]+$/.test(formData.code)) {
      Taro.showToast({
        title: "角色编码只能包含字母、数字和下划线",
        icon: "none",
      });
      return false;
    }

    return true;
  };

  // 表单提交
  const handleSubmit = async () => {
    if (!validateForm()) {
      return;
    }

    try {
      setSubmitting(true);

      if (isEdit) {
        await updateRole(Number(id), formData);
        Taro.showToast({ title: "更新成功", icon: "none" });
      } else {
        await createRole(formData);
        Taro.showToast({ title: "创建成功", icon: "none" });
      }

      // 返回列表页
      setTimeout(() => {
        Taro.navigateBack();
      }, 1000);
    } catch (error) {
      // 错误已在 hook 中处理
    } finally {
      setSubmitting(false);
    }
  };

  if (loading) {
    return <Loading>加载中...</Loading>;
  }

  return (
    <View className="role-detail-page">
      <Navbar title={isEdit ? "编辑角色" : "新增角色"}>
        <Navbar.NavLeft>
          <ArrowLeft onClick={() => Taro.navigateBack()} />
        </Navbar.NavLeft>
      </Navbar>

      <Form className="role-form" onSubmit={handleSubmit}>
        {/* 基本信息 */}
        <Cell.Group inset title="基本信息">
          <Form.Item
            name="name"
            rules={[{ required: true, message: "请输入角色名称" }]}
          >
            <Form.Label>角色名称</Form.Label>
            <Form.Control>
              <Input
                value={formData.name}
                placeholder="请输入角色名称"
                onChange={(value) => handleFieldChange("name", value)}
              />
            </Form.Control>
          </Form.Item>

          <Form.Item
            name="code"
            rules={[{ required: true, message: "请输入角色编码" }]}
          >
            <Form.Label>角色编码</Form.Label>
            <Form.Control>
              <Input
                value={formData.code}
                placeholder="请输入角色编码"
                readonly={isEdit}
                onChange={(value) => handleFieldChange("code", value)}
              />
            </Form.Control>
          </Form.Item>

          <Field name="sort" label="显示排序">
            <Stepper
              value={formData.sort}
              min={0}
              max={999}
              onChange={(value) => handleFieldChange("sort", Number(value))}
            />
          </Field>
        </Cell.Group>

        {/* 权限配置 */}
        <Cell.Group inset title="权限配置">
          <Form.Item name="status">
            <Form.Label>状态</Form.Label>
            <Form.Control>
              <Radio.Group
                value={formData.status}
                onChange={(value) => handleFieldChange("status", Number(value))}
              >
                <Radio name={1}>启用</Radio>
                <Radio name={0}>禁用</Radio>
              </Radio.Group>
            </Form.Control>
          </Form.Item>

          <Form.Item name="dataScope">
            <Form.Label>数据权限</Form.Label>
            <Form.Control>
              <Radio.Group
                value={formData.dataScope}
                onChange={(value) =>
                  handleFieldChange("dataScope", String(value))
                }
              >
                <Radio name="1">全部数据</Radio>
                <Radio name="2">部门数据</Radio>
                <Radio name="3">部门及以下数据</Radio>
                <Radio name="4">仅本人数据</Radio>
              </Radio.Group>
            </Form.Control>
          </Form.Item>
        </Cell.Group>

        {/* 表单操作 */}
        <View className="form-actions">
          <Button block color="primary" formType="submit" loading={submitting}>
            {isEdit ? "更新" : "创建"}
          </Button>
        </View>
      </Form>
    </View>
  );
};

export default RoleDetailPage;
