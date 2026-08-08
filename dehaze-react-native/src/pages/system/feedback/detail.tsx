/**
 * 反馈详情（含回复）
 */
import React, { useState, useEffect } from 'react';
import { View, Text, ScrollView, StyleSheet, TextInput, TouchableOpacity, Alert, ActivityIndicator } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';

import type { ProfileStackParamList } from '@/routes/types';
import { AppHeader } from '@/layout';
import { theme } from '@/theme';
import { FeedbackAPI } from 'dehaze-sdk-js'
import type { FeedbackDetailVO } from 'dehaze-sdk-js'

type Props = NativeStackScreenProps<ProfileStackParamList, 'SystemFeedbackDetail'>;

const SystemFeedbackDetailScreen: React.FC<Props> = ({ navigation, route }) => {
  const { feedbackId } = route.params;
  const [detail, setDetail] = useState<FeedbackDetailVO | null>(null);
  const [loading, setLoading] = useState(true);
  const [replyContent, setReplyContent] = useState('');
  const [sending, setSending] = useState(false);

  const fetchDetail = async () => {
    try {
      const data = await FeedbackAPI.getFeedbackDetail(feedbackId);
      setDetail(data);
    } catch { Alert.alert('错误', '加载失败'); }
    finally { setLoading(false); }
  };

  useEffect(() => { fetchDetail(); // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [feedbackId]);

  const handleReply = async () => {
    if (!replyContent.trim()) { Alert.alert('提示', '请输入回复内容'); return; }
    setSending(true);
    try {
      await FeedbackAPI.replyFeedback(feedbackId, { content: replyContent.trim() });
      setReplyContent('');
      fetchDetail();
    } catch { Alert.alert('错误', '回复失败'); }
    finally { setSending(false); }
  };

  if (loading) return <View style={styles.container}>
      <AppHeader title="反馈详情" showBack onBackPress={() => navigation.goBack()} /><View style={styles.centered}><ActivityIndicator size="large" color={theme.colors.primary} /></View></View>;
  if (!detail) return <View style={styles.container}>
      <AppHeader title="反馈详情" showBack onBackPress={() => navigation.goBack()} /><View style={styles.centered}><Text>无数据</Text></View></View>;

  return (
    <View style={styles.container}>
      <AppHeader title="反馈详情" showBack onBackPress={() => navigation.goBack()} />
      <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent}>
        <View style={styles.card}>
          <Text style={styles.title}>{detail.title}</Text>
          <Text style={styles.meta}>{detail.feedbackType} · {detail.username} · {detail.status}</Text>
          <Text style={styles.bodyText}>{detail.content}</Text>
        </View>
        {detail.replies?.length > 0 && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>回复记录</Text>
            {detail.replies.map((r) => (
              <View key={r.id} style={styles.replyCard}>
                <Text style={styles.replyName}>{r.replierName}</Text>
                <Text style={styles.replyContent}>{r.content}</Text>
                <Text style={styles.replyTime}>{r.createTime}</Text>
              </View>
            ))}
          </View>
        )}
        <View style={styles.replyInput}>
          <TextInput style={styles.input} value={replyContent} onChangeText={setReplyContent} placeholder="输入回复内容..." placeholderTextColor={theme.colors.text.tertiary} multiline numberOfLines={3} textAlignVertical="top" />
          <TouchableOpacity style={[styles.sendBtn, sending && styles.disabledBtn]} onPress={handleReply} disabled={sending}>
            {sending ? <ActivityIndicator size="small" color="#fff" /> : <Text style={styles.sendBtnText}>发送回复</Text>}
          </TouchableOpacity>
        </View>
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1 },
  centered: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  scrollView: { flex: 1 },
  scrollContent: { padding: theme.spacing.md, paddingBottom: theme.spacing.xxxl },
  card: { backgroundColor: theme.colors.background.primary, borderRadius: theme.layout.borderRadius.lg, padding: theme.spacing.md, marginBottom: theme.spacing.md, ...theme.layout.shadows.sm },
  title: { fontSize: theme.typography.sizes.large, fontWeight: theme.typography.weights.bold, color: theme.colors.text.primary },
  meta: { fontSize: theme.typography.sizes.tiny, color: theme.colors.text.tertiary, marginTop: 4 },
  bodyText: { fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.primary, marginTop: theme.spacing.sm, lineHeight: 22 },
  section: { marginBottom: theme.spacing.md },
  sectionTitle: { fontSize: theme.typography.sizes.medium, fontWeight: theme.typography.weights.bold, color: theme.colors.text.primary, marginBottom: theme.spacing.sm },
  replyCard: { backgroundColor: theme.colors.background.tertiary, borderRadius: theme.layout.borderRadius.sm, padding: theme.spacing.sm, marginBottom: theme.spacing.xs },
  replyName: { fontSize: theme.typography.sizes.small, fontWeight: theme.typography.weights.semibold, color: theme.colors.primary },
  replyContent: { fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.primary, marginTop: 4 },
  replyTime: { fontSize: theme.typography.sizes.tiny, color: theme.colors.text.tertiary, marginTop: 4 },
  replyInput: { marginTop: theme.spacing.md },
  input: { backgroundColor: theme.colors.background.primary, borderRadius: theme.layout.borderRadius.sm, paddingHorizontal: theme.spacing.md, paddingVertical: theme.spacing.sm, fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.primary, borderWidth: StyleSheet.hairlineWidth, borderColor: theme.colors.border.light, minHeight: 80 },
  sendBtn: { marginTop: theme.spacing.sm, backgroundColor: theme.colors.primary, borderRadius: theme.layout.borderRadius.md, paddingVertical: theme.spacing.md, alignItems: 'center' },
  disabledBtn: { opacity: 0.6 },
  sendBtnText: { fontSize: theme.typography.sizes.medium, fontWeight: theme.typography.weights.bold, color: '#fff' },
});

export default SystemFeedbackDetailScreen;
