package com.pei.dehaze.ui.login;

import android.util.Patterns;

import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.repository.RepositoryCallback;
import com.pei.dehaze.sdk.api.AuthAPI;
import com.pei.dehaze.sdk.model.auth.CaptchaResponse;
import com.pei.dehaze.sdk.model.auth.LoginRequest;
import com.pei.dehaze.sdk.model.auth.LoginResponse;
import com.pei.dehaze.sdk.utils.TokenManager;

import lombok.Getter;

@Getter
public class LoginViewModel extends ViewModel {
    private final MutableLiveData<String> username = new MutableLiveData<>("");
    private final MutableLiveData<String> password = new MutableLiveData<>("");
    private final MutableLiveData<String> captchaCode = new MutableLiveData<>("");
    private final MutableLiveData<String> captchaKey = new MutableLiveData<>("");
    private final MutableLiveData<String> captchaImage = new MutableLiveData<>("");
    private final MutableLiveData<Boolean> rememberMe = new MutableLiveData<>(true);

    private final MutableLiveData<Boolean> loading = new MutableLiveData<>(false);
    private final MutableLiveData<String> loginError = new MutableLiveData<>("");
    private final MutableLiveData<Boolean> loginSuccess = new MutableLiveData<>(false);

    public void loadCaptcha() {
        AuthAPI.getCaptcha(RepositoryAdapters.wrap(new RepositoryCallback<CaptchaResponse>() {
            @Override
            public void onSuccess(CaptchaResponse data) {
                captchaKey.postValue(data.getCaptchaKey());
                captchaImage.postValue(data.getCaptchaBase64());
            }

            @Override
            public void onError(String errorMessage) {
                loginError.postValue("获取验证码失败: " + errorMessage);
            }
        }));
    }

    public void login() {
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

        loading.setValue(true);
        loginError.setValue("");

        LoginRequest request = new LoginRequest();
        request.setUsername(username.getValue());
        request.setPassword(password.getValue());
        request.setCaptchaCode(captchaCode.getValue());
        request.setCaptchaKey(captchaKey.getValue());
        request.setRememberMe(Boolean.TRUE.equals(rememberMe.getValue()));

        AuthAPI.login(request, RepositoryAdapters.wrap(new RepositoryCallback<LoginResponse>() {
            @Override
            public void onSuccess(LoginResponse data) {
                TokenManager.setSessionId(data.getSessionId());
                loading.postValue(false);
                loginSuccess.postValue(true);
            }

            @Override
            public void onError(String errorMessage) {
                loading.postValue(false);
                loginError.postValue("登录失败: " + errorMessage);
                loadCaptcha();
            }
        }));
    }

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
