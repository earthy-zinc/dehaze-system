package com.pei.dehaze.ui.profile;

import android.content.Intent;
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
import com.pei.dehaze.ui.input.InputHistoryActivity;
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
        showLoadingState();

        profileViewModel.loadUserInfo();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (profileViewModel != null) {
            profileViewModel.loadUserInfo();
        }
    }

    private void setupListeners() {
        binding.logoutButton.setOnClickListener(v -> showLogoutConfirmDialog());
        binding.cardHistory.setOnClickListener(v ->
                startActivity(new Intent(getActivity(), InputHistoryActivity.class)));
        binding.cardNotLoggedIn.setOnClickListener(v -> navigateToLogin());
    }

    private void setupObservers() {
        profileViewModel.getUserInfo().observe(getViewLifecycleOwner(), userInfo -> {
            if (userInfo != null) {
                updateUserInfo(userInfo);
            }
        });

        profileViewModel.getLoading().observe(getViewLifecycleOwner(), isLoading -> {
            boolean loading = isLoading != null && isLoading;
            binding.progressBar.setVisibility(loading ? View.VISIBLE : View.GONE);
            if (loading) {
                showLoadingState();
            }
        });

        profileViewModel.getNotLoggedIn().observe(getViewLifecycleOwner(), notLoggedIn -> {
            if (Boolean.TRUE.equals(notLoggedIn)) {
                showNotLoggedInState();
            } else {
                showLoggedInState();
            }
        });

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
        List<String> perms = userInfo.getPerms();

        binding.tvUsername.setText(username != null && !username.isEmpty()
                ? username : "未知用户");
        binding.tvNickname.setText(nickname != null && !nickname.isEmpty()
                ? nickname : "未设置昵称");
        binding.tvAvatarInitial.setText(getInitial(username));

        String roleText = (roles == null || roles.isEmpty())
                ? "未分配角色"
                : TextUtils.join("、", roles);
        binding.tvRole.setText(roleText);

        binding.tvAccountUsername.setText(username != null && !username.isEmpty()
                ? username : "-");
        binding.tvAccountUserId.setText(userInfo.getUserId() != null
                ? String.valueOf(userInfo.getUserId()) : "-");
        binding.tvAccountNickname.setText(nickname != null && !nickname.isEmpty()
                ? nickname : "-");
        binding.tvAccountRoles.setText(roleText);

        binding.tvPermRoles.setText(roleText);
        String permText = (perms == null || perms.isEmpty())
                ? "无权限"
                : perms.size() + " 项";
        binding.tvPermPerms.setText(permText);
        binding.tvPermCreateTime.setText(userInfo.getCreateTime() != null
                && !userInfo.getCreateTime().isEmpty()
                ? userInfo.getCreateTime() : "-");
    }

    private void showLoadingState() {
        binding.tvUsername.setText("加载中...");
        binding.tvNickname.setText("加载中...");
        binding.tvAvatarInitial.setText("...");
        binding.tvRole.setText("加载中...");
        binding.tvAccountUsername.setText("加载中...");
        binding.tvAccountUserId.setText("加载中...");
        binding.tvAccountNickname.setText("加载中...");
        binding.tvAccountRoles.setText("加载中...");
        binding.tvPermRoles.setText("加载中...");
        binding.tvPermPerms.setText("加载中...");
        binding.tvPermCreateTime.setText("加载中...");
    }

    private void showNotLoggedInState() {
        binding.cardNotLoggedIn.setVisibility(View.VISIBLE);
        binding.cardUserHeader.setVisibility(View.GONE);
        binding.tvAccountSectionTitle.setVisibility(View.GONE);
        binding.cardAccountInfo.setVisibility(View.GONE);
        binding.tvPermSectionTitle.setVisibility(View.GONE);
        binding.cardPermInfo.setVisibility(View.GONE);
        binding.cardHistory.setVisibility(View.GONE);
        binding.logoutButton.setVisibility(View.GONE);
    }

    private void showLoggedInState() {
        binding.cardNotLoggedIn.setVisibility(View.GONE);
        binding.cardUserHeader.setVisibility(View.VISIBLE);
        binding.tvAccountSectionTitle.setVisibility(View.VISIBLE);
        binding.cardAccountInfo.setVisibility(View.VISIBLE);
        binding.tvPermSectionTitle.setVisibility(View.VISIBLE);
        binding.cardPermInfo.setVisibility(View.VISIBLE);
        binding.cardHistory.setVisibility(View.VISIBLE);
        binding.logoutButton.setVisibility(View.VISIBLE);
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
