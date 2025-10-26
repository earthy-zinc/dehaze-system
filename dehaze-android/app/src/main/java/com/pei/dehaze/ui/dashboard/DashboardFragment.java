package com.pei.dehaze.ui.dashboard;

import android.content.Intent;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;
import androidx.navigation.NavController;
import androidx.navigation.Navigation;

import com.pei.dehaze.R;
import com.pei.dehaze.databinding.ActivityDashboardBinding;
import com.pei.dehaze.sdk.model.user.UserInfo;
import com.pei.dehaze.ui.compare.CompareActivity;
import com.pei.dehaze.ui.evaluation.EvaluationActivity;
import com.pei.dehaze.ui.presentation.PresentationActivity;
import com.pei.dehaze.ui.dashboard.viewmodel.DashboardViewModel;

public class DashboardFragment extends Fragment {

    private DashboardViewModel dashboardViewModel;
    private ActivityDashboardBinding binding;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        binding = ActivityDashboardBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        initViewModel();
        setupObservers();
        loadData();
        setupDashboardButtons();
    }

    private void initViewModel() {
        dashboardViewModel = new ViewModelProvider(this).get(DashboardViewModel.class);
    }

    private void setupObservers() {
        dashboardViewModel.getUserInfo().observe(getViewLifecycleOwner(), userInfo -> {
            if (userInfo != null) {
                updateUI(userInfo);
            }
        });

        dashboardViewModel.getLoading().observe(getViewLifecycleOwner(), isLoading -> {
            // 处理加载状态
            binding.swipeRefresh.setRefreshing(isLoading);
        });

        dashboardViewModel.getError().observe(getViewLifecycleOwner(), errorMessage -> {
            if (errorMessage != null && !errorMessage.isEmpty()) {
                // 显示错误信息
            }
        });
        
        // 设置下拉刷新监听器
        binding.swipeRefresh.setOnRefreshListener(() -> {
            loadData();
        });
    }

    private void loadData() {
        dashboardViewModel.loadUserInfo();
    }

    private void updateUI(UserInfo userInfo) {
        // 更新UI界面
        binding.tvGreeting.setText("欢迎，" + userInfo.getUsername() + "！");
    }
    
    private void setupDashboardButtons() {
        // 设置跳转到对比模块的按钮
        binding.getRoot().findViewById(R.id.compare_module_button).setOnClickListener(v -> {
            NavController navController = Navigation.findNavController(requireActivity(), R.id.nav_host_fragment_content_main);
            navController.navigate(R.id.action_global_compareActivity);
        });
        
        // 设置跳转到评估模块的按钮
        binding.getRoot().findViewById(R.id.evaluation_module_button).setOnClickListener(v -> {
            NavController navController = Navigation.findNavController(requireActivity(), R.id.nav_host_fragment_content_main);
            navController.navigate(R.id.action_global_evaluationActivity);
        });
        
        // 设置跳转到展示模块的按钮
        binding.getRoot().findViewById(R.id.presentation_module_button).setOnClickListener(v -> {
            NavController navController = Navigation.findNavController(requireActivity(), R.id.nav_host_fragment_content_main);
            navController.navigate(R.id.action_global_presentationActivity);
        });
    }
    
    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}