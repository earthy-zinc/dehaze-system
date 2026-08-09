package com.pei.dehaze.ui.metrics_manage;

import android.os.Bundle;
import android.text.TextUtils;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.ViewModelProvider;
import androidx.viewpager2.adapter.FragmentStateAdapter;
import androidx.viewpager2.widget.ViewPager2;

import com.google.android.material.tabs.TabLayout;
import com.google.android.material.tabs.TabLayoutMediator;
import com.pei.dehaze.databinding.ActivityMetricsManageBinding;
import com.pei.dehaze.ui.metrics_manage.viewmodel.MetricsManageViewModel;
import com.pei.dehaze.ui.common.BaseActivity;
import com.pei.dehaze.utils.ToastUtils;

/**
 * 指标管理 L2 Activity
 * 评估日志列表 + 筛选 + 对比表格
 */
public class MetricsManageActivity extends BaseActivity {

    private MetricsManageViewModel viewModel;
    private ActivityMetricsManageBinding binding;
    private EvalLogListFragment evalLogFragment;
    private EvalLogListFragment predLogFragment;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityMetricsManageBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        viewModel = new ViewModelProvider(this).get(MetricsManageViewModel.class);

        initViews();
        setupObservers();
        viewModel.loadEvalLogs(null);
    }

    private void initViews() {
        binding.toolbar.setNavigationOnClickListener(v -> finish());

        binding.btnRefresh.setOnClickListener(v -> {
            String algoStr = binding.etAlgorithmFilter.getText() == null ? "" :
                    binding.etAlgorithmFilter.getText().toString().trim();
            Long algoId = TextUtils.isEmpty(algoStr) ? null : Long.parseLong(algoStr);
            evalLogFragment.setFilterAlgorithmId(algoId);
            if (predLogFragment != null) {
                predLogFragment.setFilterAlgorithmId(algoId);
            }
        });

        // ViewPager2
        binding.viewPager.setAdapter(new MetricsPagerAdapter(this));
        binding.viewPager.registerOnPageChangeCallback(new ViewPager2.OnPageChangeCallback() {
            @Override
            public void onPageSelected(int position) {
                super.onPageSelected(position);
                if (position == 0 && evalLogFragment != null) {
                    // 评估日志页
                } else if (position == 1 && predLogFragment != null) {
                    // 预测日志页
                }
            }
        });

        new TabLayoutMediator(binding.tabLayout, binding.viewPager,
                (tab, position) -> {
                    if (position == 0) tab.setText("评估日志");
                    else tab.setText("预测日志");
                }).attach();
    }

    private void setupObservers() {
        viewModel.getError().observe(this, errorMessage -> {
            if (errorMessage != null && !errorMessage.isEmpty()) {
                ToastUtils.showShort(this, errorMessage);
                viewModel.clearError();
            }
        });
    }

    private class MetricsPagerAdapter extends FragmentStateAdapter {

        MetricsPagerAdapter(@NonNull AppCompatActivity activity) {
            super(activity);
        }

        @NonNull
        @Override
        public androidx.fragment.app.Fragment createFragment(int position) {
            EvalLogListFragment fragment = new EvalLogListFragment();
            if (position == 0) {
                evalLogFragment = fragment;
            } else {
                predLogFragment = fragment;
            }
            return fragment;
        }

        @Override
        public int getItemCount() {
            return 2;
        }
    }
}
