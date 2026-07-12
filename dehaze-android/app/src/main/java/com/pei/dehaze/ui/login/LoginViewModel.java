package com.pei.dehaze.ui.login;

import android.util.Patterns;

import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.api.AuthAPI;
import com.pei.dehaze.sdk.network.ApiException;
import com.pei.dehaze.sdk.model.auth.CaptchaResponse;
import com.pei.dehaze.sdk.model.auth.LoginRequest;
import com.pei.dehaze.sdk.model.auth.LoginResponse;
import com.pei.dehaze.sdk.utils.TokenManager;

import lombok.Getter;

@Getter
public class LoginViewModel extends ViewModel {
    private final MutableLiveData<String> username = new MutableLiveData<>();
    private final MutableLiveData<String> password = new MutableLiveData<>();
    private final MutableLiveData<String> captchaCode = new MutableLiveData<>();
    private final MutableLiveData<String> captchaKey = new MutableLiveData<>();
    private final MutableLiveData<String> captchaImage = new MutableLiveData<>();

    private final MutableLiveData<Boolean> loading = new MutableLiveData<>();
    private final MutableLiveData<String> loginError = new MutableLiveData<>();
    private final MutableLiveData<Boolean> loginSuccess = new MutableLiveData<>();

    public LoginViewModel() {
        // 初始化默认值
        username.setValue("");
        password.setValue("");
        captchaCode.setValue("");
        captchaKey.setValue("");
        captchaImage.setValue("");
        loading.setValue(false);
        loginError.setValue("");
        loginSuccess.setValue(false);
    }

    /** 设置用户名（供 UI 文本变化监听器调用） */
    public void setUsername(String value) {
        username.setValue(value);
    }

    /** 设置密码（供 UI 文本变化监听器调用） */
    public void setPassword(String value) {
        password.setValue(value);
    }

    /** 设置验证码（供 UI 文本变化监听器调用） */
    public void setCaptchaCode(String value) {
        captchaCode.setValue(value);
    }

    /**
     * 获取验证码
     */
    public void loadCaptcha() {
        AuthAPI.getCaptcha(new ApiCallback<CaptchaResponse>() {
            @Override
            public void onSuccess(CaptchaResponse data) {
                captchaKey.postValue(data.getCaptchaKey());
                captchaImage.postValue(data.getCaptchaBase64());
            }

            @Override
            public void onError(String code, String message) {
                loginError.postValue("获取验证码失败: " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                loginError.postValue("网络错误: " + e.getMessage());
            }
        });
    }

    /**
     * 执行登录操作
     */
    public void login() {
        // 先进行表单验证
        if (!isUserNameValid(username.getValue())) {
            loginError.setValue("用户名格式不正确");
            return;
        }

        if (!isPasswordValid(password.getValue())) {
            loginError.setValue("密码长度不能少于6位");
            return;
        }

        if (!isCaptchaCodeValid(captchaCode.getValue())) {
            loginError.setValue("验证码不能为空");
            return;
        }

        // 显示加载状态
        loading.setValue(true);
        loginError.setValue("");

        // 构造登录请求
        LoginRequest request = new LoginRequest();
        request.setUsername(username.getValue());
        request.setPassword(password.getValue());
        request.setCaptchaCode(captchaCode.getValue());
        request.setCaptchaKey(captchaKey.getValue());

        // 发起登录请求
        AuthAPI.login(request, new ApiCallback<LoginResponse>() {
            @Override
            public void onSuccess(LoginResponse data) {
                // 关键：登录成功后保存 Token（持久化到 SharedPreferences）
                TokenManager.setToken(data.getAccessToken());
                loading.postValue(false);
                loginSuccess.postValue(true);
            }

            @Override
            public void onError(String code, String message) {
                loading.postValue(false);
                loginError.postValue("登录失败: " + message);
                // 登录失败后刷新验证码
                loadCaptcha();
            }

            @Override
            public void onFailure(ApiException e) {
                loading.postValue(false);
                loginError.postValue("网络错误: " + e.getMessage());
                // 登录失败后刷新验证码
                loadCaptcha();
            }
        });
    }

    // 表单验证方法
    private boolean isUserNameValid(String username) {
        if (username == null) {
            return false;
        }
        if (username.contains("@")) {
            return Patterns.EMAIL_ADDRESS.matcher(username).matches();
        } else {
            return !username.trim().isEmpty();
        }
    }

    private boolean isPasswordValid(String password) {
        return password != null && password.trim().length() > 5;
    }

    private boolean isCaptchaCodeValid(String captchaCode) {
        return captchaCode != null && !captchaCode.trim().isEmpty();
    }
}