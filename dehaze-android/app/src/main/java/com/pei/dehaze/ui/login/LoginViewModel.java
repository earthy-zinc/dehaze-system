package com.pei.dehaze.ui.login;

import android.util.Log;
import android.util.Patterns;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.pei.dehaze.R;
import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.api.AuthAPI;
import com.pei.dehaze.sdk.network.ApiException;
import com.pei.dehaze.sdk.model.auth.CaptchaResponse;
import com.pei.dehaze.sdk.model.auth.LoginRequest;
import com.pei.dehaze.sdk.model.auth.LoginResponse;

public class LoginViewModel extends ViewModel {

    private static final String TAG = "LoginViewModel";

    private MutableLiveData<String> username = new MutableLiveData<>();
    private MutableLiveData<String> password = new MutableLiveData<>();
    private MutableLiveData<String> captchaCode = new MutableLiveData<>();
    private MutableLiveData<String> captchaKey = new MutableLiveData<>();
    private MutableLiveData<String> captchaImage = new MutableLiveData<>();
    
    private MutableLiveData<Boolean> loading = new MutableLiveData<>();
    private MutableLiveData<String> loginError = new MutableLiveData<>();
    private MutableLiveData<Boolean> loginSuccess = new MutableLiveData<>();
    
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

    public MutableLiveData<String> getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username.setValue(username);
    }

    public MutableLiveData<String> getPassword() {
        return password;
    }

    public void setPassword(String password) {
        this.password.setValue(password);
    }

    public MutableLiveData<String> getCaptchaCode() {
        return captchaCode;
    }

    public void setCaptchaCode(String captchaCode) {
        this.captchaCode.setValue(captchaCode);
    }

    public MutableLiveData<String> getCaptchaKey() {
        return captchaKey;
    }

    public void setCaptchaKey(String captchaKey) {
        this.captchaKey.setValue(captchaKey);
    }

    public MutableLiveData<String> getCaptchaImage() {
        return captchaImage;
    }

    public MutableLiveData<Boolean> getLoading() {
        return loading;
    }

    public MutableLiveData<String> getLoginError() {
        return loginError;
    }

    public MutableLiveData<Boolean> getLoginSuccess() {
        return loginSuccess;
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
            public void onError(int code, String message) {
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
                loading.postValue(false);
                loginSuccess.postValue(true);
            }

            @Override
            public void onError(int code, String message) {
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