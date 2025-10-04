package com.pei.dehaze.activity;

import android.content.Intent;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import com.pei.dehaze.R;
import com.pei.dehaze.model.response.RegisterResponse;
import com.pei.dehaze.common.Result;
import com.pei.dehaze.user.AuthRepository;
import com.pei.dehaze.utils.InputValidator;
import com.pei.dehaze.utils.ToastUtils;

/**
 * 注册界面Activity
 */
public class RegisterActivity extends AppCompatActivity {
    private static final String TAG = "RegisterActivity";
    private EditText etUsername;
    private EditText etPassword;
    private EditText etConfirmPassword;
    private EditText etEmail;
    private EditText etNickname;
    private Button btnRegister;
    private TextView tvLogin;
    private TextView tvUsernameError;
    private TextView tvPasswordError;
    private TextView tvConfirmPasswordError;
    private TextView tvEmailError;
    private TextView tvNicknameError;
    
    private AuthRepository authRepository;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_register);

        // 初始化UI组件
        initViews();
        // 初始化Repository
        authRepository = new AuthRepository();
        // 设置监听器
        setupListeners();
    }

    /**
     * 初始化UI组件
     */
    private void initViews() {
        etUsername = findViewById(R.id.et_username);
        etPassword = findViewById(R.id.et_password);
        etConfirmPassword = findViewById(R.id.et_confirm_password);
        etEmail = findViewById(R.id.et_email);
        etNickname = findViewById(R.id.et_nickname);
        btnRegister = findViewById(R.id.btn_register);
        tvLogin = findViewById(R.id.tv_login);
        tvUsernameError = findViewById(R.id.tv_username_error);
        tvPasswordError = findViewById(R.id.tv_password_error);
        tvConfirmPasswordError = findViewById(R.id.tv_confirm_password_error);
        tvEmailError = findViewById(R.id.tv_email_error);
        tvNicknameError = findViewById(R.id.tv_nickname_error);

        // 初始禁用注册按钮
        btnRegister.setEnabled(false);
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

        // 确认密码输入监听
        etConfirmPassword.addTextChangedListener(new TextWatcher() {
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

        // 邮箱输入监听
        etEmail.addTextChangedListener(new TextWatcher() {
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

        // 昵称输入监听
        etNickname.addTextChangedListener(new TextWatcher() {
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

        // 注册按钮点击事件
        btnRegister.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                register();
            }
        });

        // 登录按钮点击事件
        tvLogin.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startActivity(new Intent(RegisterActivity.this, LoginActivity.class));
                finish();
            }
        });
    }

    /**
     * 验证输入内容
     */
    private void validateInputs() {
        String username = etUsername.getText().toString().trim();
        String password = etPassword.getText().toString().trim();
        String confirmPassword = etConfirmPassword.getText().toString().trim();
        String email = etEmail.getText().toString().trim();
        String nickname = etNickname.getText().toString().trim();

        // 验证用户名
        boolean isUsernameValid = InputValidator.isValidUsername(username);
        tvUsernameError.setVisibility(isUsernameValid ? View.GONE : View.VISIBLE);
        tvUsernameError.setText(isUsernameValid ? "" : InputValidator.getUsernameRule());

        // 验证密码
        boolean isPasswordValid = InputValidator.isValidPassword(password);
        tvPasswordError.setVisibility(isPasswordValid ? View.GONE : View.VISIBLE);
        tvPasswordError.setText(isPasswordValid ? "" : InputValidator.getPasswordRule());

        // 验证确认密码
        boolean isConfirmPasswordValid = !confirmPassword.isEmpty() && confirmPassword.equals(password);
        tvConfirmPasswordError.setVisibility(isConfirmPasswordValid ? View.GONE : View.VISIBLE);
        tvConfirmPasswordError.setText(isConfirmPasswordValid ? "" : "两次输入的密码不一致");

        // 验证邮箱
        boolean isEmailValid = InputValidator.isValidEmail(email);
        tvEmailError.setVisibility(isEmailValid ? View.GONE : View.VISIBLE);
        tvEmailError.setText(isEmailValid ? "" : InputValidator.getEmailRule());

        // 验证昵称
        boolean isNicknameValid = InputValidator.isValidNickname(nickname);
        tvNicknameError.setVisibility(isNicknameValid ? View.GONE : View.VISIBLE);
        tvNicknameError.setText(isNicknameValid ? "" : InputValidator.getNicknameRule());

        // 根据输入是否全部有效启用或禁用注册按钮
        btnRegister.setEnabled(isUsernameValid && isPasswordValid && isConfirmPasswordValid && isEmailValid && isNicknameValid);
    }

    /**
     * 执行注册操作
     */
    private void register() {
        String username = etUsername.getText().toString().trim();
        String password = etPassword.getText().toString().trim();
        String email = etEmail.getText().toString().trim();
        String nickname = etNickname.getText().toString().trim();

        // 显示加载状态
        btnRegister.setEnabled(false);
        btnRegister.setText("注册中...");

        // 调用注册接口
        authRepository.register(username, password, email, nickname, new AuthRepository.AuthCallback<Result<RegisterResponse>>() {
            @Override
            public void onSuccess(Result<RegisterResponse> response) {
                // 注册成功
                Log.d(TAG, "Register success: " + response.getMsg());
                ToastUtils.showShort(RegisterActivity.this, "注册成功，请登录");
                
                // 跳转到登录界面
                startActivity(new Intent(RegisterActivity.this, LoginActivity.class));
                finish();
            }

            @Override
            public void onError(Exception e) {
                // 注册失败
                Log.e(TAG, "Register error: " + e.getMessage());
                ToastUtils.showShort(RegisterActivity.this, e.getMessage());
                
                // 恢复按钮状态
                btnRegister.setEnabled(true);
                btnRegister.setText("注册");
            }
        });
    }
}