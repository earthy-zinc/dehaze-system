package com.pei.dehaze.ui.dashboard;

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
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.LinearLayoutManager;

import com.pei.dehaze.R;
import com.pei.dehaze.databinding.ActivityDashboardBinding;
import com.pei.dehaze.repository.DashboardRepository;
import com.pei.dehaze.sdk.model.prediction.PredictionLogVO;
import com.pei.dehaze.sdk.model.user.UserInfo;
import com.pei.dehaze.ui.common.adapter.PredictionLogAdapter;
import com.pei.dehaze.ui.dashboard.adapter.StatAdapter;
import com.pei.dehaze.ui.dashboard.viewmodel.DashboardViewModel;
import com.pei.dehaze.utils.ToastUtils;

import java.util.ArrayList;
import java.util.List;

public class DashboardFragment extends Fragment {

    private static final int[] STAT_IDS = new int[]{1, 2, 3, 4};

    private DashboardViewModel dashboardViewModel;
    private ActivityDashboardBinding binding;
    private StatAdapter statAdapter;
    private PredictionLogAdapter activityAdapter;

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

        initViews();
        initViewModel();
        setupObservers();
        loadData();
    }

    private void initViews() {
        statAdapter = new StatAdapter();
        binding.rvStatistics.setLayoutManager(new GridLayoutManager(requireContext(), 2));
        binding.rvStatistics.setAdapter(statAdapter);

        activityAdapter = new PredictionLogAdapter();
        binding.rvActivities.setLayoutManager(new LinearLayoutManager(requireContext()));
        binding.rvActivities.setAdapter(activityAdapter);

        binding.swipeRefresh.setOnRefreshListener(this::loadData);

        binding.btnPresentation.setOnClickListener(v -> navigateTo(R.id.action_global_presentationActivity));
        binding.btnCompare.setOnClickListener(v -> navigateTo(R.id.action_global_compareActivity));
        binding.btnEvaluation.setOnClickListener(v -> navigateTo(R.id.action_global_evaluationActivity));
        binding.btnDataset.setOnClickListener(v -> navigateTo(R.id.datasetFragment));
        binding.btnAlgorithm.setOnClickListener(v -> navigateTo(R.id.algorithmFragment));
    }

    private void navigateTo(int destinationId) {
        NavController navController = Navigation.findNavController(requireActivity(),
                R.id.nav_host_fragment_content_main);
        navController.navigate(destinationId);
    }

    private void initViewModel() {
        dashboardViewModel = new ViewModelProvider(this).get(DashboardViewModel.class);
    }

    private void setupObservers() {
        dashboardViewModel.getUserInfo().observe(getViewLifecycleOwner(), userInfo -> {
            if (userInfo != null) {
                updateUserInfo(userInfo);
            }
        });

        dashboardViewModel.getStats().observe(getViewLifecycleOwner(), this::updateStats);

        dashboardViewModel.getRecentActivities().observe(getViewLifecycleOwner(),
                this::updateRecentActivities);

        dashboardViewModel.getTaskTrend().observe(getViewLifecycleOwner(), trend -> {
            if (trend != null) {
                binding.chartTrend.setData(trend);
            }
        });

        dashboardViewModel.getStatusDistribution().observe(getViewLifecycleOwner(), dist -> {
            if (dist != null) {
                binding.chartStatus.setData(dist.getDistribution());
            }
        });

        dashboardViewModel.getLoading().observe(getViewLifecycleOwner(), isLoading ->
                binding.swipeRefresh.setRefreshing(isLoading != null && isLoading));

        dashboardViewModel.getError().observe(getViewLifecycleOwner(), errorMessage -> {
            if (errorMessage != null && !errorMessage.isEmpty()) {
                ToastUtils.showShort(requireContext(), errorMessage);
                dashboardViewModel.clearError();
            }
        });
    }

    private void loadData() {
        dashboardViewModel.refresh();
    }

    private void updateUserInfo(UserInfo userInfo) {
        binding.tvGreeting.setText("欢迎，" + (userInfo.getUsername() == null ? "" : userInfo.getUsername()) + "！");
        List<String> roles = userInfo.getRoles();
        String roleText = (roles == null || roles.isEmpty())
                ? "未分配角色"
                : android.text.TextUtils.join("、", roles);
        binding.tvSubtitle.setText("角色：" + roleText);
    }

    private void updateStats(DashboardRepository.StatsData stats) {
        if (stats == null) return;
        List<StatAdapter.StatItem> items = new ArrayList<>();
        items.add(new StatAdapter.StatItem(STAT_IDS[0], "数据集", (int) stats.getDatasetCount(), "已上传数据集数量"));
        items.add(new StatAdapter.StatItem(STAT_IDS[1], "算法", (int) stats.getAlgorithmCount(), "已注册算法数量"));
        items.add(new StatAdapter.StatItem(STAT_IDS[2], "任务", (int) stats.getTaskCount(), "处理任务总数"));
        items.add(new StatAdapter.StatItem(STAT_IDS[3], "历史记录", (int) stats.getHistoryCount(), "去雾历史记录数"));
        statAdapter.submitList(items);
    }

    private void updateRecentActivities(List<PredictionLogVO> activities) {
        activityAdapter.submitList(activities);
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}
