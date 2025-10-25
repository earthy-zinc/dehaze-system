package com.pei.dehaze.ui.login;

import android.os.Bundle;
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

import com.bumptech.glide.Glide;
import com.google.android.material.button.MaterialButton;
import com.google.android.material.switchmaterial.SwitchMaterial;
import com.google.android.material.textfield.TextInputEditText;
import com.pei.dehaze.R;
import com.pei.dehaze.databinding.FragmentLoginBinding;

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
                // TODO: 跳转到主界面
            }
        });
        
        // 观察验证码图片
        loginViewModel.getCaptchaImage().observe(getViewLifecycleOwner(), base64Image -> {
            if (!base64Image.isEmpty()) {
                // 解码Base64图片并显示
                byte[] decodedString = Base64.decode(base64Image, Base64.DEFAULT);
                Glide.with(this)
                        .asBitmap()
                        .load(decodedString)
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