import React, { useState } from "react";
import { View } from "@tarojs/components";
import Taro, { useRouter, useLoad } from "@tarojs/taro";
import {
  Navbar,
  Form,
  Input,
  Radio,
  Button,
  Loading,
  Cell,
  Picker,
  Popup,
  Field,
} from "@taroify/core";
import { ArrowLeft } from "@taroify/icons";
import { useUserManagement } from "@/hooks/useUserManagement";
import { useDeptManagement } from "@/hooks/useDeptManagement";
import { useRoleManagement } from "@/hooks/useRoleManagement";
import type { UserForm } from "dehaze-sdk-js";
import "./detail.scss";

const UserDetailPage: React.FC = () => {
  const router = useRouter();
  const { id } = router.params;
  const isEdit = !!id;

  const { createUser, updateUser, getUserDetail } = useUserManagement();
  const { deptOptions, fetchDeptOptions } = useDeptManagement();
  const { getRoleOptions } = useRoleManagement();

  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [openDeptPicker, setOpenDeptPicker] = useState(false);
  const [openRolePicker, setOpenRolePicker] = useState(false);
  const [roleOptions, setRoleOptions] = useState<any[]>([]);
  const [formData, setFormData] = useState<UserForm>({
    username: "",
    nickname: "",
    mobile: "",
    email: "",
    gender: 0,
    status: 1,
    deptId: undefined,
    roleIds: [],
  });

  // 页面加载时初始化数据
  useLoad(async () => {
    try {
      // 并行获取部门选项和角色选项
      await Promise.all([
        fetchDeptOptions(),
        getRoleOptions().then((roles) => setRoleOptions(roles)),
      ]);

      // 如果是编辑模式，获取用户详情
      if (isEdit) {
        await loadUserData();
      }
    } catch (error) {
      console.error("初始化数据失败:", error);
      Taro.showToast({ title: "初始化数据失败", icon: "none" });
    }
  });

  // 加载用户数据
  const loadUserData = async () => {
    if (!id) return;

    try {
      setLoading(true);
      const userData = await getUserDetail(Number(id));
      setFormData({
        username: userData.username || "",
        nickname: userData.nickname || "",
        mobile: userData.mobile || "",
        email: userData.email || "",
        gender: userData.gender ?? 0,
        status: userData.status ?? 1,
        deptId: userData.deptId,
        roleIds: userData.roleIds || [],
      });
    } catch (error) {
      Taro.showToast({ title: "获取用户信息失败", icon: "none" });
      console.error("获取用户信息失败:", error);
      // 加载失败时返回上一页
      setTimeout(() => {
        Taro.navigateBack();
      }, 1500);
    } finally {
      setLoading(false);
    }
  };

  // 表单字段更新
  const handleFieldChange = (field: keyof UserForm, value: any) => {
    setFormData((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  // 表单提交
  const handleSubmit = async () => {
    // 表单验证
    const trimmedUsername = formData.username?.trim();
    if (!trimmedUsername) {
      Taro.showToast({ title: "请输入用户名", icon: "none" });
      return;
    }

    const trimmedNickname = formData.nickname?.trim();
    if (!trimmedNickname) {
      Taro.showToast({ title: "请输入用户昵称", icon: "none" });
      return;
    }

    if (formData.email && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) {
      Taro.showToast({ title: "请输入正确的邮箱地址", icon: "none" });
      return;
    }

    if (formData.mobile && !/^1[3-9]\d{9}$/.test(formData.mobile)) {
      Taro.showToast({ title: "请输入正确的手机号码", icon: "none" });
      return;
    }

    if (!formData.roleIds || formData.roleIds.length === 0) {
      Taro.showToast({ title: "请选择用户角色", icon: "none" });
      return;
    }

    try {
      setSubmitting(true);

      // 准备提交数据，去除空格
      const submitData = {
        ...formData,
        username: trimmedUsername,
        nickname: trimmedNickname,
      };

      if (isEdit) {
        await updateUser(Number(id), submitData);
        Taro.showToast({ title: "更新成功", icon: "none" });
      } else {
        await createUser(submitData);
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
    <View className="user-detail-page">
      <Navbar title={isEdit ? "编辑用户" : "新增用户"}>
        <Navbar.NavLeft>
          <ArrowLeft onClick={() => Taro.navigateBack()} />
        </Navbar.NavLeft>
      </Navbar>

      <Form className="user-form" onSubmit={handleSubmit}>
        {/* 基本信息 */}
        <Cell.Group inset title="基本信息">
          <Form.Item
            name="username"
            rules={[{ required: true, message: "请输入用户名" }]}
          >
            <Form.Label>用户名</Form.Label>
            <Form.Control>
              <Input
                value={formData.username}
                readonly={isEdit}
                placeholder="请输入用户名"
                onChange={(value) => handleFieldChange("username", value)}
              />
            </Form.Control>
          </Form.Item>

          <Form.Item
            name="nickname"
            rules={[{ required: true, message: "请输入用户昵称" }]}
          >
            <Form.Label>用户昵称</Form.Label>
            <Form.Control>
              <Input
                value={formData.nickname}
                placeholder="请输入用户昵称"
                onChange={(value) => handleFieldChange("nickname", value)}
              />
            </Form.Control>
          </Form.Item>

          <Form.Item name="gender">
            <Form.Label>性别</Form.Label>
            <Form.Control>
              <Radio.Group
                value={formData.gender}
                onChange={(value) => handleFieldChange("gender", Number(value))}
              >
                <Radio name={0}>未知</Radio>
                <Radio name={1}>男</Radio>
                <Radio name={2}>女</Radio>
              </Radio.Group>
            </Form.Control>
          </Form.Item>
        </Cell.Group>

        {/* 组织信息 */}
        <Cell.Group inset title="组织信息">
          <Field
            name="deptId"
            label="所属部门"
            isLink
            onClick={() => setOpenDeptPicker(true)}
          >
            <Input
              readonly
              placeholder="请选择所属部门"
              value={
                formData.deptId
                  ? deptOptions?.find((dept) => dept.value === formData.deptId)
                      ?.label
                  : ""
              }
            />
          </Field>

          <Field
            name="roleIds"
            label="用户角色"
            isLink
            required
            onClick={() => setOpenRolePicker(true)}
          >
            <Input
              readonly
              placeholder="请选择用户角色"
              value={
                formData.roleIds && formData.roleIds.length > 0
                  ? formData.roleIds
                      .map(
                        (roleId) =>
                          roleOptions?.find(
                            (role: any) => role.value === roleId
                          )?.label
                      )
                      .filter(Boolean)
                      .join("、")
                  : ""
              }
            />
          </Field>

          <Form.Item name="status">
            <Form.Label>状态</Form.Label>
            <Form.Control>
              <Radio.Group
                value={formData.status}
                onChange={(value) => handleFieldChange("status", Number(value))}
              >
                <Radio name={1}>正常</Radio>
                <Radio name={0}>禁用</Radio>
              </Radio.Group>
            </Form.Control>
          </Form.Item>
        </Cell.Group>

        {/* 联系方式 */}
        <Cell.Group inset title="联系方式">
          <Form.Item name="mobile">
            <Form.Label>手机号码</Form.Label>
            <Form.Control>
              <Input
                value={formData.mobile}
                placeholder="请输入手机号码"
                type="number"
                maxlength={11}
                onChange={(value) => handleFieldChange("mobile", value)}
              />
            </Form.Control>
          </Form.Item>

          <Form.Item name="email">
            <Form.Label>邮箱</Form.Label>
            <Form.Control>
              <Input
                value={formData.email}
                placeholder="请输入邮箱"
                onChange={(value) => handleFieldChange("email", value)}
              />
            </Form.Control>
          </Form.Item>
        </Cell.Group>

        {/* 表单操作 */}
        <View className="form-actions">
          <Button block color="primary" formType="submit" loading={submitting}>
            {isEdit ? "更新" : "创建"}
          </Button>
        </View>

        {/* 部门选择弹窗 */}
        <Popup
          open={openDeptPicker}
          rounded
          placement="bottom"
          onClose={setOpenDeptPicker}
        >
          <Popup.Backdrop />
          <Picker
            title="选择所属部门"
            cancelText="取消"
            confirmText="确认"
            columns={
              deptOptions?.map((dept) => ({
                text: dept.label,
                value: String(dept.value),
              })) || []
            }
            onCancel={() => setOpenDeptPicker(false)}
            onConfirm={(values) => {
              handleFieldChange("deptId", Number(values[0]));
              setOpenDeptPicker(false);
            }}
          />
        </Popup>

        {/* 角色选择弹窗 */}
        <Popup
          open={openRolePicker}
          rounded
          placement="bottom"
          onClose={setOpenRolePicker}
        >
          <Popup.Backdrop />
          <Picker
            title="选择用户角色"
            cancelText="取消"
            confirmText="确认"
            columns={
              roleOptions?.map((role) => ({
                text: role.label,
                value: String(role.value),
              })) || []
            }
            onCancel={() => setOpenRolePicker(false)}
            onConfirm={(values) => {
              // 单选模式，如果需要多选可能需要使用其他组件或自定义实现
              handleFieldChange("roleIds", [Number(values[0])]);
              setOpenRolePicker(false);
            }}
          />
        </Popup>
      </Form>
    </View>
  );
};

export default UserDetailPage;
