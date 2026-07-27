package com.pei.dehaze.ui.register;

import android.util.Patterns;

import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.repository.RepositoryCallback;
import com.pei.dehaze.sdk.api.AuthAPI;
import com.pei.dehaze.sdk.model.auth.CaptchaResponse;
import com.pei.dehaze.sdk.model.auth.LoginRequest;
import com.pei.dehaze.sdk.model.auth.LoginResponse;

import lombok.Getter;

@Getter
public class RegisterViewModel extends ViewModel {
    private final MutableLiveData<String> username = new MutableLiveData<>("");
    private final MutableLiveData<String> nickname = new MutableLiveData<>("");
    private final MutableLiveData<String> password = new MutableLiveData<>("");
    private final MutableLiveData<String> confirmPassword = new MutableLiveData<>("");
    private final MutableLiveData<String> captchaCode = new MutableLiveData<>("");
    private final MutableLiveData<String> captchaKey = new MutableLiveData<>("");
    private final MutableLiveData<String> captchaImage = new MutableLiveData<>("");

    private final MutableLiveData<Boolean> loading = new MutableLiveData<>(false);
    private final MutableLiveData<String> error = new MutableLiveData<>("");
    private final MutableLiveData<Boolean> registerSuccess = new MutableLiveData<>(false);

    public void loadCaptcha() {
        AuthAPI.getCaptcha(RepositoryAdapters.wrap(new RepositoryCallback<CaptchaResponse>() {
            @Override
            public void onSuccess(CaptchaResponse data) {
                captchaKey.postValue(data.getCaptchaKey());
                captchaImage.postValue(data.getCaptchaBase64());
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue("获取验证码失败: " + errorMessage);
            }
        }));
    }

    public void register() {
        if (!isUserNameValid(username.getValue())) {
            error.setValue("用户名格式不正确");
            return;
        }

        if (!isPasswordValid(password.getValue())) {
            error.setValue("密码长度不能少于6位");
            return;
        }

        if (password.getValue() == null || !password.getValue().equals(confirmPassword.getValue())) {
            error.setValue("两次密码不一致");
            return;
        }

        if (!isCaptchaCodeValid(captchaCode.getValue())) {
            error.setValue("验证码不能为空");
            return;
        }

        loading.setValue(true);
        error.setValue("");

        LoginRequest request = new LoginRequest();
        request.setUsername(username.getValue());
        request.setPassword(password.getValue());
        request.setNickname(nickname.getValue());
        request.setCaptchaCode(captchaCode.getValue());
        request.setCaptchaKey(captchaKey.getValue());

        AuthAPI.register(request, RepositoryAdapters.wrap(new RepositoryCallback<LoginResponse>() {
            @Override
            public void onSuccess(LoginResponse data) {
                loading.postValue(false);
                registerSuccess.postValue(true);
            }

            @Override
            public void onError(String errorMessage) {
                loading.postValue(false);
                error.postValue("注册失败: " + errorMessage);
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
