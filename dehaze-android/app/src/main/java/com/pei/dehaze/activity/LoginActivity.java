package com.pei.dehaze.activity;

import android.content.Intent;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import com.bumptech.glide.Glide;
import com.pei.dehaze.MainActivity;
import com.pei.dehaze.R;
import com.pei.dehaze.common.ResultCode;
import com.pei.dehaze.model.response.LoginResponse;
import com.pei.dehaze.model.response.CaptchaResponse;
import com.pei.dehaze.common.Result;
import com.pei.dehaze.user.AuthRepository;
import com.pei.dehaze.user.UserManager;
import com.pei.dehaze.utils.ToastUtils;
import com.pei.dehaze.network.ApiService;
import com.pei.dehaze.network.RetrofitClient;

/**
 * 登录界面Activity
 */
public class LoginActivity extends AppCompatActivity {
    private static final String TAG = "LoginActivity";
    private EditText etUsername;
    private EditText etPassword;
    private EditText etCaptchaCode;
    private ImageView ivCaptcha;
    private Button btnLogin;
    private TextView tvRegister;
    private TextView tvForgotPassword;
    private TextView tvUsernameError;
    private TextView tvPasswordError;
    private TextView tvCaptchaError;
    
    private AuthRepository authRepository;
    private ApiService apiService;
    private String captchaKey;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_login);

        // 初始化UserManager
        UserManager.init(this);

        // 检查用户是否已登录，如果已登录则直接跳转到主界面
        if (UserManager.getInstance().isLoggedIn()) {
            Log.d(TAG, "User is already logged in, redirecting to MainActivity");
            startActivity(new Intent(this, MainActivity.class));
            finish();
            return;
        }

        // 初始化UI组件
        initViews();
        // 初始化Repository
        authRepository = new AuthRepository();
        // 初始化API服务
        apiService = RetrofitClient.getInstance().getApiService();
        // 设置监听器
        setupListeners();
        // 获取验证码
        fetchCaptcha();
    }

    /**
     * 初始化UI组件
     */
    private void initViews() {
        etUsername = findViewById(R.id.et_username);
        etPassword = findViewById(R.id.et_password);
        etCaptchaCode = findViewById(R.id.et_captcha_code);
        ivCaptcha = findViewById(R.id.iv_captcha);
        btnLogin = findViewById(R.id.btn_login);
        tvRegister = findViewById(R.id.tv_register);
        tvForgotPassword = findViewById(R.id.tv_forgot_password);
        tvUsernameError = findViewById(R.id.tv_username_error);
        tvPasswordError = findViewById(R.id.tv_password_error);
        tvCaptchaError = findViewById(R.id.tv_captcha_error);

        // 初始禁用登录按钮
        btnLogin.setEnabled(false);
    }

    /**
     * 设置监听器
     */
    private void setupListeners() {
        // 用户名输入监听
        etUsername.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {
            }

            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {
                validateInputs();
            }

            @Override
            public void afterTextChanged(Editable s) {
            }
        });

        // 密码输入监听
        etPassword.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {
            }

            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {
                validateInputs();
            }

            @Override
            public void afterTextChanged(Editable s) {
            }
        });

        // 验证码输入监听
        etCaptchaCode.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {
            }

            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {
                validateInputs();
            }

            @Override
            public void afterTextChanged(Editable s) {
            }
        });

        // 验证码图片点击事件
        ivCaptcha.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                fetchCaptcha();
            }
        });

        // 登录按钮点击事件
        btnLogin.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                login();
            }
        });

        // 注册按钮点击事件
        tvRegister.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startActivity(new Intent(LoginActivity.this, RegisterActivity.class));
            }
        });

        // 忘记密码点击事件
        tvForgotPassword.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                ToastUtils.showShort(LoginActivity.this, "忘记密码功能暂未实现");
            }
        });
    }

    /**
     * 验证输入内容
     */
    private void validateInputs() {
        String username = etUsername.getText().toString().trim();
        String password = etPassword.getText().toString().trim();
        String captchaCode = etCaptchaCode.getText().toString().trim();

        boolean isUsernameValid = !username.isEmpty();
        boolean isPasswordValid = !password.isEmpty();
        boolean isCaptchaValid = !captchaCode.isEmpty();

        // 显示或隐藏用户名错误提示
        tvUsernameError.setVisibility(isUsernameValid ? View.GONE : View.VISIBLE);
        // 显示或隐藏密码错误提示
        tvPasswordError.setVisibility(isPasswordValid ? View.GONE : View.VISIBLE);
        // 显示或隐藏验证码错误提示
        tvCaptchaError.setVisibility(isCaptchaValid ? View.GONE : View.VISIBLE);

        // 根据输入是否有效启用或禁用登录按钮
        btnLogin.setEnabled(isUsernameValid && isPasswordValid && isCaptchaValid);
    }

    /**
     * 获取验证码
     */
    private void fetchCaptcha() {
        apiService.getCaptcha().enqueue(new retrofit2.Callback<Result<CaptchaResponse>>() {
            @Override
            public void onResponse(retrofit2.Call<Result<CaptchaResponse>> call, retrofit2.Response<Result<CaptchaResponse>> response) {
                if (response.isSuccessful() && response.body() != null) {
                    Result<CaptchaResponse> res = response.body();
                    if (res.getCode().equals(ResultCode.SUCCESS.getCode())) {
                        captchaKey = res.getData().getCaptchaKey();
                        String captchaBase64 = res.getData().getCaptchaBase64();
                        
                        // 使用Glide加载验证码图片
                        Glide.with(LoginActivity.this)
                                .load(captchaBase64)
                                .into(ivCaptcha);
                    } else {
                        ToastUtils.showShort(LoginActivity.this, "获取验证码失败: " + res.getMsg());
                    }
                } else {
                    ToastUtils.showShort(LoginActivity.this, "获取验证码失败，请稍后重试");
                }
            }

            @Override
            public void onFailure(retrofit2.Call<Result<CaptchaResponse>> call, Throwable t) {
                Log.e(TAG, "获取验证码网络错误", t);
                ToastUtils.showShort(LoginActivity.this, "网络连接失败，请检查网络设置");
            }
        });
    }

    /**
     * 执行登录操作
     */
    private void login() {
        String username = etUsername.getText().toString().trim();
        String password = etPassword.getText().toString().trim();
        String captchaCode = etCaptchaCode.getText().toString().trim();

        // 显示加载状态
        btnLogin.setEnabled(false);
        btnLogin.setText("登录中...");

        // 调用登录接口
        authRepository.login(username, password, captchaCode, captchaKey, new AuthRepository.AuthCallback<Result<LoginResponse>>() {
            @Override
            public void onSuccess(Result<LoginResponse> res) {
                // 登录成功
                Log.d(TAG, "Login success: " + res.getMsg());
                ToastUtils.showShort(LoginActivity.this, "登录成功");
                
                // 跳转到主界面
                startActivity(new Intent(LoginActivity.this, MainActivity.class));
                finish();
            }

            @Override
            public void onError(Exception e) {
                // 登录失败
                Log.e(TAG, "Login error: " + e.getMessage());
                ToastUtils.showShort(LoginActivity.this, e.getMessage());
                
                // 恢复按钮状态
                btnLogin.setEnabled(true);
                btnLogin.setText("登录");
                
                // 如果是验证码错误，重新获取验证码
                if (e.getMessage().contains("验证码")) {
                    fetchCaptcha();
                    etCaptchaCode.setText("");
                }
            }
        });
    }
}