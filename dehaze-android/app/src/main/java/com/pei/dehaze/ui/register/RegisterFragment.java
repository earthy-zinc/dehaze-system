package com.pei.dehaze.ui.register;

import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;
import androidx.navigation.Navigation;

import com.bumptech.glide.Glide;
import com.pei.dehaze.R;
import com.pei.dehaze.databinding.FragmentRegisterBinding;
import com.pei.dehaze.utils.ToastUtils;

public class RegisterFragment extends Fragment {

    private RegisterViewModel registerViewModel;
    private FragmentRegisterBinding binding;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        binding = FragmentRegisterBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        registerViewModel = new ViewModelProvider(this).get(RegisterViewModel.class);

        binding.setViewModel(registerViewModel);
        binding.setLifecycleOwner(this);

        setupUI();
        registerViewModel.loadCaptcha();
    }

    private void setupUI() {
        binding.registerButton.setOnClickListener(v -> registerViewModel.register());
        binding.captchaImage.setOnClickListener(v -> registerViewModel.loadCaptcha());
        binding.loginLink.setOnClickListener(v ->
                Navigation.findNavController(requireView()).navigate(R.id.action_register_to_login));

        binding.usernameEditText.addTextChangedListener(new SimpleTextWatcher() {
            @Override
            public void afterTextChanged(Editable s) {
                registerViewModel.getUsername().setValue(s.toString());
            }
        });
        binding.nicknameEditText.addTextChangedListener(new SimpleTextWatcher() {
            @Override
            public void afterTextChanged(Editable s) {
                registerViewModel.getNickname().setValue(s.toString());
            }
        });
        binding.passwordEditText.addTextChangedListener(new SimpleTextWatcher() {
            @Override
            public void afterTextChanged(Editable s) {
                registerViewModel.getPassword().setValue(s.toString());
            }
        });
        binding.confirmPasswordEditText.addTextChangedListener(new SimpleTextWatcher() {
            @Override
            public void afterTextChanged(Editable s) {
                registerViewModel.getConfirmPassword().setValue(s.toString());
            }
        });
        binding.captchaEditText.addTextChangedListener(new SimpleTextWatcher() {
            @Override
            public void afterTextChanged(Editable s) {
                registerViewModel.getCaptchaCode().setValue(s.toString());
            }
        });

        registerViewModel.getError().observe(getViewLifecycleOwner(), error -> {
            if (error != null && !error.isEmpty()) {
                ToastUtils.showLong(getContext(), error);
            }
        });

        registerViewModel.getLoading().observe(getViewLifecycleOwner(), isLoading -> {
            boolean loading = Boolean.TRUE.equals(isLoading);
            binding.registerButton.setEnabled(!loading);
            binding.registerProgressBar.setVisibility(loading ? View.VISIBLE : View.GONE);
            binding.registerButton.setText(loading ? "注册中..." : "注 册");
        });

        registerViewModel.getRegisterSuccess().observe(getViewLifecycleOwner(), success -> {
            if (Boolean.TRUE.equals(success)) {
                ToastUtils.showShort(getContext(), "注册成功，请登录");
                Navigation.findNavController(requireView()).navigate(R.id.action_register_to_login);
            }
        });

        registerViewModel.getCaptchaImage().observe(getViewLifecycleOwner(), base64Image -> {
            if (base64Image != null && !base64Image.isEmpty()) {
                String base64String = base64Image;
                if (base64Image.startsWith("data:image")) {
                    base64String = base64Image.substring(base64Image.indexOf(",") + 1);
                }
                String base64ImageData = "data:image/png;base64," + base64String;
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

    private static abstract class SimpleTextWatcher implements TextWatcher {
        @Override
        public void beforeTextChanged(CharSequence s, int start, int count, int after) {
        }

        @Override
        public void onTextChanged(CharSequence s, int start, int before, int count) {
        }
    }
}
