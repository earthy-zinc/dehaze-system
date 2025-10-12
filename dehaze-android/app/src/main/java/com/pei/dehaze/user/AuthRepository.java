package com.pei.dehaze.user;


import androidx.annotation.NonNull;

import com.pei.dehaze.model.response.LoginResponse;
import com.pei.dehaze.model.request.RegisterRequest;
import com.pei.dehaze.model.response.RegisterResponse;
import com.pei.dehaze.common.Result;
import com.pei.dehaze.network.ApiService;
import com.pei.dehaze.network.RetrofitClient;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import timber.log.Timber;

/**
 * 认证仓库类，负责处理用户登录、注册等认证相关的业务逻辑
 */
public class AuthRepository {

    private final ApiService apiService;

    public AuthRepository() {
        // 获取API服务实例
        apiService = RetrofitClient.getInstance().getApiService();
    }

    /**
     * 用户登录方法
     */
    public void login(String username, String password, String captchaCode, String captchaKey, final AuthCallback<Result<LoginResponse>> callback) {
        Timber.d("Login attempt with username: %s", username);

        // 发送登录请求
        Call<Result<LoginResponse>> call = apiService.login(username, password, captchaCode, captchaKey);
        call.enqueue(new Callback<Result<LoginResponse>>() {
            @Override
            public void onResponse(@NonNull Call<Result<LoginResponse>> call, @NonNull Response<Result<LoginResponse>> response) {
                if (response.isSuccessful() && response.body() != null) {
                    Result<LoginResponse> res = response.body();
                    if ("00000".equals(res.getCode())) {
                        // 登录成功
                        Timber.d("Login successful: %s", res.getMsg());
                        // 保存用户信息
                        UserManager.getInstance().saveLoginInfo(res);
                        callback.onSuccess(res);
                    } else {
                        // 登录失败，返回错误信息
                        Timber.e("Login failed: %s", res.getMsg());
                        callback.onError(new Exception(res.getMsg()));
                    }
                } else {
                    // 服务器返回错误
                    String errorMessage = "登录失败，请检查网络连接或稍后重试";
                    try {
                        if (response.errorBody() != null) {
                            errorMessage = response.errorBody().string();
                        }
                    } catch (Exception e) {
                        Timber.e(e, "Error parsing error response");
                    }
                    Timber.e("Login error: %s", errorMessage);
                    callback.onError(new Exception(errorMessage));
                }
            }

            @Override
            public void onFailure(Call<Result<LoginResponse>> call, Throwable t) {
                // 网络请求失败
                Timber.e(t, "Login network error");
                callback.onError(new Exception("网络连接失败，请检查网络设置"));
            }
        });
    }

    /**
     * 用户注册方法
     */
    public void register(String username, String password, String email, String nickname, final AuthCallback<Result<RegisterResponse>> callback) {
        Timber.d("Register attempt with username: %s", username);
        
        // 创建注册请求
        RegisterRequest registerRequest = new RegisterRequest(username, password, email, nickname);
        
        // 发送注册请求
        Call<Result<RegisterResponse>> call = apiService.register(registerRequest);
        call.enqueue(new Callback<Result<RegisterResponse>>() {
            @Override
            public void onResponse(@NonNull Call<Result<RegisterResponse>> call, @NonNull Response<Result<RegisterResponse>> res) {
                if (res.isSuccessful() && res.body() != null) {
                    Result<RegisterResponse> registerResponse = res.body();
                    if ("00000".equals(registerResponse.getCode())) {
                        // 注册成功
                        Timber.d("Register successful: %s", registerResponse.getMsg());
                        callback.onSuccess(registerResponse);
                    } else {
                        // 注册失败，返回错误信息
                        Timber.e("Register failed: %s", registerResponse.getMsg());
                        callback.onError(new Exception(registerResponse.getMsg()));
                    }
                } else {
                    // 服务器返回错误
                    String errorMessage = "注册失败，请检查网络连接或稍后重试";
                    try {
                        if (res.errorBody() != null) {
                            errorMessage = res.errorBody().string();
                        }
                    } catch (Exception e) {
                        Timber.e(e, "Error parsing error res");
                    }
                    Timber.e("Register error: %s", errorMessage);
                    callback.onError(new Exception(errorMessage));
                }
            }

            @Override
            public void onFailure(Call<Result<RegisterResponse>> call, Throwable t) {
                // 网络请求失败
                Timber.e(t, "Register network error");
                callback.onError(new Exception("网络连接失败，请检查网络设置"));
            }
        });
    }

    /**
     * 认证回调接口
     */
    public interface AuthCallback<T> {
        void onSuccess(T response);
        void onError(Exception e);
    }
}