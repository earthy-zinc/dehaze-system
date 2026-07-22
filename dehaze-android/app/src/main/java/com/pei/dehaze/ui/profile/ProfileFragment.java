package com.pei.dehaze.ui.profile;

import android.os.Bundle;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;
import androidx.navigation.NavController;
import androidx.navigation.NavOptions;
import androidx.navigation.Navigation;

import com.pei.dehaze.R;
import com.pei.dehaze.databinding.FragmentProfileBinding;
import com.pei.dehaze.sdk.model.user.UserInfo;
import com.pei.dehaze.ui.profile.viewmodel.ProfileViewModel;
import com.pei.dehaze.utils.ToastUtils;

import java.util.List;

public class ProfileFragment extends Fragment {

    private FragmentProfileBinding binding;
    private ProfileViewModel profileViewModel;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        binding = FragmentProfileBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        profileViewModel = new ViewModelProvider(this).get(ProfileViewModel.class);

        setupListeners();
        setupObservers();

        profileViewModel.loadUserInfo();
    }

    private void setupListeners() {
        binding.logoutButton.setOnClickListener(v -> showLogoutConfirmDialog());
    }

    private void setupObservers() {
        profileViewModel.getUserInfo().observe(getViewLifecycleOwner(), userInfo -> {
            if (userInfo != null) {
                updateUserInfo(userInfo);
            } else {
                showEmptyState();
            }
        });

        profileViewModel.getLoading().observe(getViewLifecycleOwner(), isLoading ->
                binding.progressBar.setVisibility(
                        isLoading != null && isLoading ? View.VISIBLE : View.GONE));

        profileViewModel.getError().observe(getViewLifecycleOwner(), errorMessage -> {
            if (errorMessage != null && !errorMessage.isEmpty()) {
                ToastUtils.showShort(requireContext(), errorMessage);
                profileViewModel.clearError();
            }
        });

        profileViewModel.getLogoutSuccess().observe(getViewLifecycleOwner(), success -> {
            if (success != null && success) {
                navigateToLogin();
            }
        });
    }

    private void updateUserInfo(UserInfo userInfo) {
        String username = userInfo.getUsername();
        String nickname = userInfo.getNickname();
        List<String> roles = userInfo.getRoles();

        binding.tvUsername.setText(username != null && !username.isEmpty()
                ? username : "未知用户");
        binding.tvNickname.setText(nickname != null && !nickname.isEmpty()
                ? nickname : "未设置昵称");
        binding.tvAvatarInitial.setText(getInitial(username));

        String roleText = (roles == null || roles.isEmpty())
                ? "未分配角色"
                : TextUtils.join("、", roles);
        binding.tvRole.setText(roleText);

        // 账号信息卡片
        binding.tvAccountUsername.setText(username != null && !username.isEmpty()
                ? username : "-");
        binding.tvAccountUserId.setText(userInfo.getUserId() != null
                ? String.valueOf(userInfo.getUserId()) : "-");
        binding.tvAccountNickname.setText(nickname != null && !nickname.isEmpty()
                ? nickname : "-");
        binding.tvAccountRoles.setText(roleText);
    }

    private void showEmptyState() {
        binding.tvUsername.setText("未登录");
        binding.tvNickname.setText("无法获取用户信息");
        binding.tvAvatarInitial.setText("?");
        binding.tvRole.setText("-");
        binding.tvAccountUsername.setText("-");
        binding.tvAccountUserId.setText("-");
        binding.tvAccountNickname.setText("-");
        binding.tvAccountRoles.setText("-");
    }

    private String getInitial(String text) {
        if (text == null || text.isEmpty()) {
            return "?";
        }
        return String.valueOf(text.charAt(0)).toUpperCase();
    }

    private void showLogoutConfirmDialog() {
        new AlertDialog.Builder(requireContext())
                .setTitle("退出登录")
                .setMessage("确定要退出当前账号吗？")
                .setPositiveButton("确定", (dialog, which) -> profileViewModel.logout())
                .setNegativeButton("取消", null)
                .show();
    }

    private void navigateToLogin() {
        NavController navController = Navigation.findNavController(requireActivity(),
                R.id.nav_host_fragment_content_main);
        // 清除整个回退栈，以登录页作为新的根
        NavOptions options = new NavOptions.Builder()
                .setPopUpTo(R.id.nav_graph, true)
                .build();
        navController.navigate(R.id.loginFragment, null, options);
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}
