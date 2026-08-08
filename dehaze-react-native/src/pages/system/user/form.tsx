/**
 * 用户表单（新增/编辑）
 */
import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
} from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';

import type { ProfileStackParamList } from '@/routes/types';
import { AppHeader } from '@/layout';
import { theme } from '@/theme';
import { UserAPI } from 'dehaze-sdk-js'
import type { UserForm } from 'dehaze-sdk-js'

type Props = NativeStackScreenProps<ProfileStackParamList, 'SystemUserForm'>;

const SystemUserFormScreen: React.FC<Props> = ({ navigation, route }) => {
  const userId = route.params?.userId;
  const isEdit = !!userId;
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [form, setForm] = useState<UserForm>({
    username: '',
    nickname: '',
    email: '',
    mobile: '',
    gender: 0,
    status: 1,
  });

  useEffect(() => {
    if (isEdit && userId) {
      setLoading(true);
      UserAPI.getFormData(userId)
        .then((data) => setForm(data))
        .catch(() => Alert.alert('错误', '加载用户信息失败'))
        .finally(() => setLoading(false));
    }
  }, [isEdit, userId]);

  const handleSave = async () => {
    if (!form.username?.trim()) {
      Alert.alert('提示', '请输入用户名');
      return;
    }
    setSaving(true);
    try {
      if (isEdit) {
        await UserAPI.update(userId!, form);
      } else {
        await UserAPI.add(form);
      }
      navigation.goBack();
    } catch {
      Alert.alert('错误', '保存失败');
    } finally {
      setSaving(false);
    }
  };

  const updateField = (field: keyof UserForm, value: any) => {
    setForm((prev) => ({ ...prev, [field]: value }));
  };

  if (loading) {
    return (
      <View style={styles.container}>
      <AppHeader title={isEdit ? '编辑用户' : '新增用户'} showBack onBackPress={() => navigation.goBack()} />
        <View style={styles.loadingWrap}>
          <ActivityIndicator size="large" color={theme.colors.primary} />
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <AppHeader title={isEdit ? '编辑用户' : '新增用户'} showBack onBackPress={() => navigation.goBack()} />
      <ScrollView style={styles.container} contentContainerStyle={styles.content}>
        <FormField label="用户名" required>
          <TextInput
            style={styles.input}
            value={form.username}
            onChangeText={(v) => updateField('username', v)}
            placeholder="请输入用户名"
            placeholderTextColor={theme.colors.text.tertiary}
            editable={!isEdit}
          />
        </FormField>
        <FormField label="昵称" required>
          <TextInput
            style={styles.input}
            value={form.nickname}
            onChangeText={(v) => updateField('nickname', v)}
            placeholder="请输入昵称"
            placeholderTextColor={theme.colors.text.tertiary}
          />
        </FormField>
        <FormField label="邮箱">
          <TextInput
            style={styles.input}
            value={form.email}
            onChangeText={(v) => updateField('email', v)}
            placeholder="请输入邮箱"
            placeholderTextColor={theme.colors.text.tertiary}
            keyboardType="email-address"
          />
        </FormField>
        <FormField label="手机号">
          <TextInput
            style={styles.input}
            value={form.mobile}
            onChangeText={(v) => updateField('mobile', v)}
            placeholder="请输入手机号"
            placeholderTextColor={theme.colors.text.tertiary}
            keyboardType="phone-pad"
          />
        </FormField>
        <FormField label="状态">
          <View style={styles.toggleRow}>
            <TouchableOpacity
              style={[styles.toggleBtn, form.status === 1 && styles.toggleBtnActive]}
              onPress={() => updateField('status', 1)}
            >
              <Text style={[styles.toggleText, form.status === 1 && styles.toggleTextActive]}>启用</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.toggleBtn, form.status === 0 && styles.toggleBtnActive]}
              onPress={() => updateField('status', 0)}
            >
              <Text style={[styles.toggleText, form.status === 0 && styles.toggleTextActive]}>禁用</Text>
            </TouchableOpacity>
          </View>
        </FormField>

        <TouchableOpacity
          style={[styles.saveBtn, saving && styles.saveBtnDisabled]}
          onPress={handleSave}
          disabled={saving}
          activeOpacity={0.7}
        >
          {saving ? (
            <ActivityIndicator size="small" color="#fff" />
          ) : (
            <Text style={styles.saveBtnText}>保存</Text>
          )}
        </TouchableOpacity>
      </ScrollView>
    </View>
  );
};

const FormField: React.FC<{ label: string; required?: boolean; children: React.ReactNode }> = ({
  label,
  required,
  children,
}) => (
  <View style={styles.field}>
    <Text style={styles.fieldLabel}>
      {label}
      {required && <Text style={styles.required}> *</Text>}
    </Text>
    {children}
  </View>
);

const styles = StyleSheet.create({
  container: { flex: 1 },
  content: { padding: theme.spacing.md, paddingBottom: theme.spacing.xxxl },
  loadingWrap: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  field: { marginBottom: theme.spacing.md },
  fieldLabel: {
    fontSize: theme.typography.sizes.bodySmall,
    fontWeight: theme.typography.weights.semibold,
    color: theme.colors.text.primary,
    marginBottom: theme.spacing.xs,
  },
  required: { color: theme.colors.status.error },
  input: {
    backgroundColor: theme.colors.background.primary,
    borderRadius: theme.layout.borderRadius.sm,
    paddingHorizontal: theme.spacing.md,
    paddingVertical: theme.spacing.sm,
    fontSize: theme.typography.sizes.bodySmall,
    color: theme.colors.text.primary,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: theme.colors.border.light,
  },
  toggleRow: { flexDirection: 'row', gap: theme.spacing.sm },
  toggleBtn: {
    flex: 1,
    paddingVertical: theme.spacing.sm,
    borderRadius: theme.layout.borderRadius.sm,
    backgroundColor: theme.colors.background.tertiary,
    alignItems: 'center',
  },
  toggleBtnActive: { backgroundColor: theme.colors.primary },
  toggleText: { fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.secondary },
  toggleTextActive: { color: '#fff', fontWeight: theme.typography.weights.semibold },
  saveBtn: {
    marginTop: theme.spacing.lg,
    backgroundColor: theme.colors.primary,
    borderRadius: theme.layout.borderRadius.md,
    paddingVertical: theme.spacing.md,
    alignItems: 'center',
  },
  saveBtnDisabled: { opacity: 0.6 },
  saveBtnText: {
    fontSize: theme.typography.sizes.medium,
    fontWeight: theme.typography.weights.bold,
    color: '#fff',
  },
});

export default SystemUserFormScreen;
