/**
 * 群发消息
 */
import React, { useState } from 'react';
import { View, Text, ScrollView, StyleSheet, TextInput, TouchableOpacity, Alert, ActivityIndicator } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';

import type { ProfileStackParamList } from '@/routes/types';
import { AppHeader } from '@/layout';
import { theme } from '@/theme';
import { MessageAPI } from 'dehaze-sdk-js';

type Props = NativeStackScreenProps<ProfileStackParamList, 'SystemMessageSend'>;

const SystemMessageSendScreen: React.FC<Props> = ({ navigation }) => {
  const [form, setForm] = useState({ type: 'system', title: '', content: '', recipientIdsStr: '' });
  const [sending, setSending] = useState(false);

  const handleSend = async () => {
    if (!form.title.trim() || !form.content.trim()) { Alert.alert('提示', '标题和内容必填'); return; }
    if (!form.recipientIdsStr.trim()) { Alert.alert('提示', '请输入接收用户ID（逗号分隔）'); return; }
    const recipientIds = form.recipientIdsStr.split(',').map(Number).filter(Boolean);
    if (recipientIds.length === 0) { Alert.alert('提示', '请输入有效的用户ID'); return; }
    setSending(true);
    try {
      await MessageAPI.send({ type: form.type, title: form.title.trim(), content: form.content.trim(), recipientIds });
      Alert.alert('成功', '消息已发送');
      navigation.goBack();
    } catch { Alert.alert('错误', '发送失败'); }
    finally { setSending(false); }
  };

  return (
    <View style={st.container}>
      <AppHeader title="群发消息" showBack onBackPress={() => navigation.goBack()} />
      <ScrollView style={st.scrollView} contentContainerStyle={st.scrollContent}>
        <F label="消息类型"><TextInput style={st.input} value={form.type} onChangeText={(v) => setForm((p) => ({ ...p, type: v }))} placeholder="system" placeholderTextColor={theme.colors.text.tertiary} /></F>
        <F label="标题" required><TextInput style={st.input} value={form.title} onChangeText={(v) => setForm((p) => ({ ...p, title: v }))} placeholder="消息标题" placeholderTextColor={theme.colors.text.tertiary} /></F>
        <F label="内容" required><TextInput style={[st.input, st.textareaInput]} value={form.content} onChangeText={(v) => setForm((p) => ({ ...p, content: v }))} placeholder="消息内容" placeholderTextColor={theme.colors.text.tertiary} multiline textAlignVertical="top" /></F>
        <F label="接收用户ID（逗号分隔）" required><TextInput style={st.input} value={form.recipientIdsStr} onChangeText={(v) => setForm((p) => ({ ...p, recipientIdsStr: v }))} placeholder="1,2,3" placeholderTextColor={theme.colors.text.tertiary} keyboardType="numeric" /></F>
        <TouchableOpacity style={[st.sendBtn, sending && st.disabledBtn]} onPress={handleSend} disabled={sending}>
          {sending ? <ActivityIndicator size="small" color="#fff" /> : <Text style={st.sendBtnText}>发送消息</Text>}
        </TouchableOpacity>
      </ScrollView>
    </View>
  );
};

const F: React.FC<{ label: string; required?: boolean; children: React.ReactNode }> = ({ label, required, children }) => (
  <View style={st.fieldWrapper}>
    <Text style={st.fieldLabel}>{label}{required && <Text style={st.fieldRequired}> *</Text>}</Text>
    {children}
  </View>
);

const st = StyleSheet.create({
  container: { flex: 1 },
  scrollView: { flex: 1 },
  scrollContent: { padding: theme.spacing.md, paddingBottom: theme.spacing.xxxl },
  fieldWrapper: { marginBottom: theme.spacing.md },
  fieldLabel: { fontSize: theme.typography.sizes.bodySmall, fontWeight: theme.typography.weights.semibold, color: theme.colors.text.primary, marginBottom: theme.spacing.xs },
  fieldRequired: { color: theme.colors.status.error },
  input: { backgroundColor: theme.colors.background.primary, borderRadius: theme.layout.borderRadius.sm, paddingHorizontal: theme.spacing.md, paddingVertical: theme.spacing.sm, fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.primary, borderWidth: StyleSheet.hairlineWidth, borderColor: theme.colors.border.light },
  textareaInput: { minHeight: 100 },
  sendBtn: { marginTop: theme.spacing.lg, backgroundColor: theme.colors.primary, borderRadius: theme.layout.borderRadius.md, paddingVertical: theme.spacing.md, alignItems: 'center' },
  disabledBtn: { opacity: 0.6 },
  sendBtnText: { fontSize: theme.typography.sizes.medium, fontWeight: theme.typography.weights.bold, color: '#fff' },
});

export default SystemMessageSendScreen;
