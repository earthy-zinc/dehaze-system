/**
 * 登录页
 *
 * 对接 dehaze-sdk-js AuthAPI：
 * - 登录（用户名 + 密码 + 图形验证码）
 * - 获取验证码图片
 * - 登录成功后由 AuthContext 跳转 Home
 */
import { useAuth } from '@/store';
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

const LoginScreen: React.FC = () => {
  const { login } = useAuth();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
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

  const handleLogin = async () => {
    if (!username || !password) {
      Alert.alert('提示', '请输入用户名和密码');
      return;
    }
    if (captcha && !captchaCode) {
      Alert.alert('提示', '请输入验证码');
      return;
    }

    setLoading(true);
    try {
      await login({
        username,
        password,
        captchaKey: captcha?.captchaKey,
        captchaCode: captcha ? captchaCode : undefined,
      });
    } catch (e: any) {
      Alert.alert('登录失败', e?.message || '用户名或密码错误');
      loadCaptcha();
      setCaptchaCode('');
    } finally {
      setLoading(false);
    }
  };

  return (
    <LinearGradient
      colors={['#3B82F6', '#6366F1']}
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
              {/* 品牌头部 */}
              <View style={styles.header}>
                <View style={styles.logoContainer}>
                  <LinearGradient
                    colors={['#3B82F6', '#6366F1']}
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 1 }}
                    style={styles.logoGradient}
                  >
                    <Text style={styles.logo}>雾</Text>
                  </LinearGradient>
                </View>
                <Text style={styles.title}>图像去雾系统</Text>
                <Text style={styles.subtitle}>
                  Professional Image Dehaze Platform
                </Text>
              </View>

              {/* 表单 */}
              <View style={styles.form}>
                {/* 用户名 */}
                <View style={styles.inputGroup}>
                  <Text style={styles.label}>用户名</Text>
                  <View style={styles.inputWrap}>
                    <Ionicons
                      name="person-outline"
                      size={18}
                      color="#9ca3af"
                      style={styles.inputIcon}
                    />
                    <TextInput
                      style={styles.input}
                      placeholder="请输入用户名"
                      placeholderTextColor="#9ca3af"
                      value={username}
                      onChangeText={setUsername}
                      autoCapitalize="none"
                      returnKeyType="next"
                    />
                  </View>
                </View>

                {/* 密码 */}
                <View style={styles.inputGroup}>
                  <Text style={styles.label}>密码</Text>
                  <View style={styles.inputWrap}>
                    <Ionicons
                      name="lock-closed-outline"
                      size={18}
                      color="#9ca3af"
                      style={styles.inputIcon}
                    />
                    <TextInput
                      style={styles.input}
                      placeholder="请输入密码"
                      placeholderTextColor="#9ca3af"
                      value={password}
                      onChangeText={setPassword}
                      secureTextEntry={!showPassword}
                      returnKeyType={captcha ? 'next' : 'done'}
                    />
                    <TouchableOpacity
                      onPress={() => setShowPassword(s => !s)}
                      style={styles.eyeBtn}
                      hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
                    >
                      <Ionicons
                        name={showPassword ? 'eye-outline' : 'eye-off-outline'}
                        size={18}
                        color="#9ca3af"
                      />
                    </TouchableOpacity>
                  </View>
                </View>

                {/* 验证码 */}
                {captcha && (
                  <View style={styles.inputGroup}>
                    <Text style={styles.label}>验证码</Text>
                    <View style={styles.captchaRow}>
                      <View style={[styles.inputWrap, styles.captchaInputWrap]}>
                        <Ionicons
                          name="shield-checkmark-outline"
                          size={18}
                          color="#9ca3af"
                          style={styles.inputIcon}
                        />
                        <TextInput
                          style={styles.input}
                          placeholder="请输入验证码"
                          placeholderTextColor="#9ca3af"
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
                          <ActivityIndicator color="#3B82F6" />
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

                {/* 登录按钮 */}
                <TouchableOpacity
                  style={styles.buttonWrap}
                  onPress={handleLogin}
                  disabled={loading}
                  activeOpacity={0.9}
                >
                  <LinearGradient
                    colors={loading ? ['#93c5fd', '#a5b4fc'] : ['#3B82F6', '#6366F1']}
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 0 }}
                    style={styles.button}
                  >
                    {loading ? (
                      <ActivityIndicator color="white" />
                    ) : (
                      <>
                        <Text style={styles.buttonText}>登录</Text>
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
              </View>
            </View>

            <View style={styles.footer}>
              <Text style={[styles.footerText, styles.footerTextSpacing]}>
                Copyright © 2022 - 2024 DehazeSystem All Rights Reserved.
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
    paddingVertical: 24,
    paddingHorizontal: 20,
  },
  card: {
    width: '100%',
    maxWidth: 420,
    backgroundColor: 'white',
    borderRadius: 24,
    padding: 28,
    shadowColor: '#1e3a8a',
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
    marginBottom: 16,
  },
  logoGradient: {
    width: 68,
    height: 68,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#3B82F6',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.4,
    shadowRadius: 12,
    elevation: 6,
  },
  logo: {
    fontSize: 34,
    fontWeight: 'bold',
    color: 'white',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1f2937',
    letterSpacing: -0.5,
    marginBottom: 6,
  },
  subtitle: {
    fontSize: 13,
    color: '#9ca3af',
    letterSpacing: 0.5,
  },
  form: {
    width: '100%',
  },
  inputGroup: {
    marginBottom: 18,
  },
  label: {
    fontSize: 13,
    fontWeight: '600',
    color: '#374151',
    marginBottom: 8,
    marginLeft: 4,
  },
  inputWrap: {
    flexDirection: 'row',
    alignItems: 'center',
    height: 52,
    borderWidth: 1.5,
    borderColor: '#e5e7eb',
    borderRadius: 14,
    paddingHorizontal: 14,
    backgroundColor: '#f9fafb',
  },
  inputIcon: {
    marginRight: 10,
  },
  input: {
    flex: 1,
    fontSize: 16,
    color: '#1f2937',
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
    borderColor: '#e5e7eb',
    backgroundColor: '#f9fafb',
    overflow: 'hidden',
  },
  captchaImage: {
    width: 100,
    height: 40,
  },
  buttonWrap: {
    marginTop: 8,
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
    color: 'white',
    fontSize: 17,
    fontWeight: '700',
    letterSpacing: 1,
  },
  buttonIcon: {
    marginLeft: 8,
  },
  footer: {
    marginTop: 28,
    alignItems: 'center',
  },
  footerText: {
    fontSize: 12,
    color: 'rgba(255, 255, 255, 0.75)',
  },
  footerTextSpacing: {
    marginTop: 4,
  },
});

export default LoginScreen;
