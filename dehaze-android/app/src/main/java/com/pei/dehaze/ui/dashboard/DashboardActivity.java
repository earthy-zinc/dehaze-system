package com.pei.dehaze.ui.dashboard;

import android.os.Bundle;

import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.ViewModelProvider;

import com.pei.dehaze.R;
import com.pei.dehaze.sdk.model.user.UserInfo;
import com.pei.dehaze.ui.dashboard.viewmodel.DashboardViewModel;

public class DashboardActivity extends AppCompatActivity {

    private DashboardViewModel dashboardViewModel;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_dashboard);

        initViewModel();
        setupObservers();
        loadData();
    }

    private void initViewModel() {
        dashboardViewModel = new ViewModelProvider(this).get(DashboardViewModel.class);
    }

    private void setupObservers() {
        dashboardViewModel.getUserInfo().observe(this, userInfo -> {
            if (userInfo != null) {
                updateUI(userInfo);
            }
        });

        dashboardViewModel.getLoading().observe(this, isLoading -> {
            // 处理加载状态
        });

        dashboardViewModel.getError().observe(this, errorMessage -> {
            if (errorMessage != null && !errorMessage.isEmpty()) {
                // 显示错误信息
            }
        });
    }

    private void loadData() {
        dashboardViewModel.loadUserInfo();
    }

    private void updateUI(UserInfo userInfo) {
        // 更新UI界面
    }
}