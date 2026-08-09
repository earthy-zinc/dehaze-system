/**
 * 注册页
 *
 * 对接 dehaze-sdk-js AuthAPI.register：
 * - 用户名 + 昵称 + 密码 + 确认密码 + 图形验证码
 * - 注册成功后自动登录（后端返回 sessionId），跳转 Home
 */
import { AuthAPI } from 'dehaze-sdk-js';
import type { CaptchaResult } from 'dehaze-sdk-js';
import React, { useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Image,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import LinearGradient from 'react-native-linear-gradient';
import Ionicons from 'react-native-vector-icons/Ionicons';
import { colors } from '@/theme/colors';
import { useNavigation } from '@react-navigation/native';
import { spacing, layout } from '@/theme/spacing';

const RegisterScreen: React.FC = () => {
  const navigation = useNavigation<any>();
  const [username, setUsername] = useState('');
  const [nickname, setNickname] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [captchaCode, setCaptchaCode] = useState('');
  const [captcha, setCaptcha] = useState<CaptchaResult | null>(null);
  const [captchaLoading, setCaptchaLoading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  const loadCaptcha = async () => {
    setCaptchaLoading(true);
    try {
      const result = await AuthAPI.getCaptcha();
      const base64 = result.captchaBase64.startsWith('data:')
        ? result.captchaBase64
        : `data:image/png;base64,${result.captchaBase64}`;
      setCaptcha({ ...result, captchaBase64: base64 });
    } catch {
      setCaptcha(null);
    } finally {
      setCaptchaLoading(false);
    }
  };

  useEffect(() => {
    loadCaptcha();
  }, []);

  const handleRegister = async () => {
    if (!username.trim()) {
      Alert.alert('提示', '请输入用户名');
      return;
    }
    if (!nickname.trim()) {
      Alert.alert('提示', '请输入昵称');
      return;
    }
    if (!password) {
      Alert.alert('提示', '请输入密码');
      return;
    }
    if (password !== confirmPassword) {
      Alert.alert('提示', '两次密码不一致');
      return;
    }
    if (captcha && !captchaCode) {
      Alert.alert('提示', '请输入验证码');
      return;
    }

    setLoading(true);
    try {
      await AuthAPI.register({
        username: username.trim(),
        password,
        nickname: nickname.trim(),
        captchaKey: captcha?.captchaKey,
        captchaCode: captcha ? captchaCode : undefined,
      });
      Alert.alert('注册成功', '请使用新账号登录', [
        { text: '去登录', onPress: () => navigation.replace('Login') },
      ]);
    } catch (e: unknown) {
      const err = e as { message?: string };
      Alert.alert('注册失败', err?.message || '注册失败，请稍后重试');
      loadCaptcha();
      setCaptchaCode('');
    } finally {
      setLoading(false);
    }
  };

  const inputGroups: {
     label: string;
     icon: string;
     value: string;
     onChange: (v: string) => void;
     placeholder: string;
     secure: boolean;
   }[] = [
    {
      label: '用户名',
      icon: 'person-outline',
      value: username,
      onChange: setUsername,
      placeholder: '请输入用户名',
      secure: false,
    },
    {
      label: '昵称',
      icon: 'happy-outline',
      value: nickname,
      onChange: setNickname,
      placeholder: '请输入昵称',
      secure: false,
    },
    {
      label: '密码',
      icon: 'lock-closed-outline',
      value: password,
      onChange: setPassword,
      placeholder: '请输入密码',
      secure: !showPassword,
    },
    {
      label: '确认密码',
      icon: 'lock-closed-outline',
      value: confirmPassword,
      onChange: setConfirmPassword,
      placeholder: '请再次输入密码',
      secure: !showPassword,
    },
  ];

  return (
    <LinearGradient
      colors={[colors.primary, colors.gradient.primary[1]]}
      start={{ x: 0, y: 0 }}
      end={{ x: 1, y: 1 }}
      style={styles.gradient}
    >
      <SafeAreaView style={styles.container} edges={['top', 'bottom']}>
        <KeyboardAvoidingView
          behavior={Platform.OS === 'ios' ? 'padding' : undefined}
          style={styles.flex}
        >
          <ScrollView
            contentContainerStyle={styles.scrollContainer}
            keyboardShouldPersistTaps="handled"
          >
            <View style={styles.card}>
              <View style={styles.header}>
                <View style={styles.logoContainer}>
                  <LinearGradient
                    colors={[colors.primary, colors.gradient.primary[1]]}
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 1 }}
                    style={styles.logoGradient}
                  >
                    <Text style={styles.logo}>雾</Text>
                  </LinearGradient>
                </View>
                <Text style={styles.title}>用户注册</Text>
                <Text style={styles.subtitle}>Create Your Dehaze Account</Text>
              </View>

              <View style={styles.form}>
                {inputGroups.map((group, index) => (
                  <View key={index} style={styles.inputGroup}>
                    <Text style={styles.label}>{group.label}</Text>
                    <View style={styles.inputWrap}>
                      <Ionicons
                        name={group.icon}
                        size={18}
                        color={colors.text.tertiary}
                        style={styles.inputIcon}
                      />
                      <TextInput
                        style={styles.input}
                        placeholder={group.placeholder}
                        placeholderTextColor={colors.text.tertiary}
                        value={group.value}
                        onChangeText={group.onChange}
                        secureTextEntry={group.secure}
                        autoCapitalize="none"
                        returnKeyType="next"
                      />
                      {(group.label === '密码' || group.label === '确认密码') && (
                        <TouchableOpacity
                          onPress={() => setShowPassword(s => !s)}
                          style={styles.eyeBtn}
                          hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
                        >
                          <Ionicons
                            name={showPassword ? 'eye-outline' : 'eye-off-outline'}
                            size={18}
                            color={colors.text.tertiary}
                          />
                        </TouchableOpacity>
                      )}
                    </View>
                  </View>
                ))}

                {captcha && (
                  <View style={styles.inputGroup}>
                    <Text style={styles.label}>验证码</Text>
                    <View style={styles.captchaRow}>
                      <View style={[styles.inputWrap, styles.captchaInputWrap]}>
                        <Ionicons
                          name="shield-checkmark-outline"
                          size={18}
                          color={colors.text.tertiary}
                          style={styles.inputIcon}
                        />
                        <TextInput
                          style={styles.input}
                          placeholder="请输入验证码"
                          placeholderTextColor={colors.text.tertiary}
                          value={captchaCode}
                          onChangeText={setCaptchaCode}
                          autoCapitalize="none"
                          returnKeyType="done"
                        />
                      </View>
                      <TouchableOpacity
                        style={styles.captchaContainer}
                        onPress={loadCaptcha}
                        disabled={captchaLoading}
                        activeOpacity={0.8}
                      >
                        {captchaLoading ? (
                          <ActivityIndicator color={colors.primary} />
                        ) : (
                          <Image
                            style={styles.captchaImage}
                            source={{ uri: captcha.captchaBase64 }}
                            resizeMode="contain"
                          />
                        )}
                      </TouchableOpacity>
                    </View>
                  </View>
                )}

                <TouchableOpacity
                  style={styles.buttonWrap}
                  onPress={handleRegister}
                  disabled={loading}
                  activeOpacity={0.9}
                >
                  <LinearGradient
                    colors={loading ? ['#93c5fd', '#a5b4fc'] : [colors.primary, colors.gradient.primary[1]]}
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 0 }}
                    style={styles.button}
                  >
                    {loading ? (
                      <ActivityIndicator color="white" />
                    ) : (
                      <>
                        <Text style={styles.buttonText}>注册</Text>
                        <Ionicons
                          name="arrow-forward"
                          size={18}
                          color="white"
                          style={styles.buttonIcon}
                        />
                      </>
                    )}
                  </LinearGradient>
                </TouchableOpacity>

                <TouchableOpacity
                  style={styles.registerLink}
                  onPress={() => navigation.replace('Login')}
                  activeOpacity={0.7}
                >
                  <Text style={styles.registerLinkText}>已有账号？立即登录</Text>
                </TouchableOpacity>
              </View>
            </View>

            <View style={styles.footer}>
              <Text style={[styles.footerText, styles.footerTextSpacing]}>
                {`Copyright © 2022 - ${new Date().getFullYear()} DehazeSystem All Rights Reserved.`}
              </Text>
            </View>
          </ScrollView>
        </KeyboardAvoidingView>
      </SafeAreaView>
    </LinearGradient>
  );
};

const styles = StyleSheet.create({
  gradient: {
    flex: 1,
  },
  flex: {
    flex: 1,
  },
  container: {
    flex: 1,
  },
  scrollContainer: {
    flexGrow: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: spacing.lg,
    paddingHorizontal: 20,
  },
  card: {
    width: '100%',
    maxWidth: 420,
    backgroundColor: colors.background.primary,
    borderRadius: layout.borderRadius.xxl,
    padding: 28,
    shadowColor: colors.primaryDark,
    shadowOffset: { width: 0, height: 12 },
    shadowOpacity: 0.25,
    shadowRadius: 24,
    elevation: 8,
  },
  header: {
    alignItems: 'center',
    marginBottom: 28,
  },
  logoContainer: {
    marginBottom: spacing.md,
  },
  logoGradient: {
    width: 68,
    height: 68,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: colors.primary,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.4,
    shadowRadius: 12,
    elevation: 6,
  },
  logo: {
    fontSize: 34,
    fontWeight: 'bold',
    color: colors.text.inverse,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: colors.text.primary,
    letterSpacing: -0.5,
    marginBottom: 6,
  },
  subtitle: {
    fontSize: 13,
    color: colors.text.tertiary,
    letterSpacing: 0.5,
  },
  form: {
    width: '100%',
  },
  inputGroup: {
    marginBottom: spacing.md,
  },
  label: {
    fontSize: 13,
    fontWeight: '600',
    color: colors.text.primary,
    marginBottom: spacing.sm,
    marginLeft: 4,
  },
  inputWrap: {
    flexDirection: 'row',
    alignItems: 'center',
    height: 52,
    borderWidth: 1.5,
    borderColor: colors.border.light,
    borderRadius: 14,
    paddingHorizontal: 14,
    backgroundColor: colors.background.secondary,
  },
  inputIcon: {
    marginRight: 10,
  },
  input: {
    flex: 1,
    fontSize: 16,
    color: colors.text.primary,
    paddingVertical: 0,
  },
  eyeBtn: {
    padding: 4,
  },
  captchaRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  captchaInputWrap: {
    flex: 1,
  },
  captchaContainer: {
    justifyContent: 'center',
    alignItems: 'center',
    width: 110,
    height: 52,
    borderRadius: 14,
    borderWidth: 1.5,
    borderColor: colors.border.light,
    backgroundColor: colors.background.secondary,
    overflow: 'hidden',
  },
  captchaImage: {
    width: 100,
    height: 40,
  },
  buttonWrap: {
    marginTop: spacing.sm,
    borderRadius: 14,
    overflow: 'hidden',
  },
  button: {
    height: 54,
    borderRadius: 14,
    justifyContent: 'center',
    alignItems: 'center',
    flexDirection: 'row',
  },
  buttonText: {
    color: colors.text.inverse,
    fontSize: 17,
    fontWeight: '700',
    letterSpacing: 1,
  },
  buttonIcon: {
    marginLeft: spacing.sm,
  },
  registerLink: {
    marginTop: spacing.md,
    alignItems: 'center',
  },
  registerLinkText: {
    fontSize: 14,
    color: colors.primary,
    fontWeight: '600',
  },
  footer: {
    marginTop: 28,
    alignItems: 'center',
  },
  footerText: {
    fontSize: 12,
    color: colors.background.translucent,
  },
  footerTextSpacing: {
    marginTop: 4,
  },
});

export default RegisterScreen;
