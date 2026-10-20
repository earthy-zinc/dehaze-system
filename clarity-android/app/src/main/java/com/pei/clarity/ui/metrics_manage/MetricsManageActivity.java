package com.pei.clarity.ui.metrics_manage;

import android.os.Bundle;
import android.text.TextUtils;

import androidx.lifecycle.ViewModelProvider;

import com.pei.clarity.databinding.ActivityMetricsManageBinding;
import com.pei.clarity.ui.metrics_manage.viewmodel.MetricsManageViewModel;
import com.pei.clarity.ui.common.BaseActivity;
import com.pei.clarity.utils.ToastUtils;

/**
 * 指标管理 L2 Activity
 * 评估日志列表 + 模型筛选 + 对比表格
 */
public class MetricsManageActivity extends BaseActivity {

    private MetricsManageViewModel viewModel;
    private ActivityMetricsManageBinding binding;
    private EvalLogListFragment evalLogFragment;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityMetricsManageBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        viewModel = new ViewModelProvider(this).get(MetricsManageViewModel.class);

        initViews();
        setupObservers();

        int containerId = binding.fragmentContainer.getId();
        if (savedInstanceState == null) {
            evalLogFragment = new EvalLogListFragment();
            getSupportFragmentManager().beginTransaction()
                    .replace(containerId, evalLogFragment)
                    .commit();
        } else {
            evalLogFragment = (EvalLogListFragment) getSupportFragmentManager()
                    .findFragmentById(containerId);
        }

        viewModel.loadEvalLogs(null);
    }

    private void initViews() {
        binding.toolbar.setNavigationOnClickListener(v -> finish());

        binding.btnRefresh.setOnClickListener(v -> {
            String filter = binding.etModelFilter.getText() == null ? ""
                    : binding.etModelFilter.getText().toString().trim();
            if (evalLogFragment != null) {
                evalLogFragment.setFilterModelId(TextUtils.isEmpty(filter) ? null : filter);
            }
        });
    }

    private void setupObservers() {
        viewModel.getError().observe(this, errorMessage -> {
            if (errorMessage != null && !errorMessage.isEmpty()) {
                ToastUtils.showShort(this, errorMessage);
                viewModel.clearError();
            }
        });
    }
}
