package com.pei.dehaze.ui.login;

import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.util.Base64;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.CompoundButton;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatDelegate;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;
import androidx.navigation.Navigation;

import com.bumptech.glide.Glide;
import com.google.android.material.button.MaterialButton;
import com.google.android.material.switchmaterial.SwitchMaterial;
import com.google.android.material.textfield.TextInputEditText;
import com.pei.dehaze.R;
import com.pei.dehaze.databinding.FragmentLoginBinding;

import timber.log.Timber;

public class LoginFragment extends Fragment {

    private LoginViewModel loginViewModel;
    private FragmentLoginBinding binding;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        binding = FragmentLoginBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        
        // 初始化 ViewModel
        loginViewModel = new ViewModelProvider(this).get(LoginViewModel.class);
        
        // 设置数据绑定
        binding.setViewModel(loginViewModel);
        binding.setLifecycleOwner(this);
        
        // 初始化界面
        setupUI();
        
        // 获取验证码
        loginViewModel.loadCaptcha();
    }
    
    private void setupUI() {
        // 登录按钮点击事件
        binding.loginButton.setOnClickListener(v -> loginViewModel.login());
        
        // 验证码图片点击事件（刷新验证码）
        binding.captchaImage.setOnClickListener(v -> loginViewModel.loadCaptcha());
        
        // 添加文本变化监听器以更新ViewModel
        binding.usernameEditText.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {
            }

            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {
            }

            @Override
            public void afterTextChanged(Editable s) {
                loginViewModel.setUsername(s.toString());
            }
        });
        
        binding.passwordEditText.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {
            }

            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {
            }

            @Override
            public void afterTextChanged(Editable s) {
                loginViewModel.setPassword(s.toString());
            }
        });
        
        binding.captchaEditText.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {
            }

            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {
            }

            @Override
            public void afterTextChanged(Editable s) {
                loginViewModel.setCaptchaCode(s.toString());
            }
        });
        
        // 主题切换
        binding.themeSwitch.setChecked(AppCompatDelegate.getDefaultNightMode() == AppCompatDelegate.MODE_NIGHT_YES);
        binding.themeSwitch.setOnCheckedChangeListener((buttonView, isChecked) -> {
            if (isChecked) {
                AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_YES);
            } else {
                AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_NO);
            }
        });
        
        // 观察登录错误信息
        loginViewModel.getLoginError().observe(getViewLifecycleOwner(), error -> {
            if (!error.isEmpty()) {
                Toast.makeText(getContext(), error, Toast.LENGTH_LONG).show();
            }
        });
        
        // 观察登录成功状态
        loginViewModel.getLoginSuccess().observe(getViewLifecycleOwner(), success -> {
            if (success) {
                // 登录成功，跳转到主界面
                Toast.makeText(getContext(), "登录成功", Toast.LENGTH_SHORT).show();
                // 使用 popUpTo 和 inclusive 清除登录页面的回退栈
                Navigation.findNavController(requireView()).navigate(
                    R.id.action_login_to_dashboard,
                    null,
                    null,
                    null
                );
            }
        });
        
        // 观察验证码图片
        loginViewModel.getCaptchaImage().observe(getViewLifecycleOwner(), base64Image -> {
            Timber.d("验证码图片: %s", base64Image);
            if (!base64Image.isEmpty()) {
                // 处理带前缀的Base64图片数据
                String base64String = base64Image;
                // 如果包含前缀，则提取真正的Base64编码部分
                if (base64Image.startsWith("data:image")) {
                    base64String = base64Image.substring(base64Image.indexOf(",") + 1);
                }
                
                // 使用Glide加载Base64图片
                String base64ImageData = "data:image/png;base64," + base64String;
                Timber.d("Base64图片数据: %s", base64ImageData);
                Glide.with(this)
                        .load(base64ImageData)
                        .into(binding.captchaImage);
            }
        });
    }
    
    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}