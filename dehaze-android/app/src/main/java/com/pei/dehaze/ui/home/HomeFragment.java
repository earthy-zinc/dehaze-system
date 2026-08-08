package com.pei.dehaze.ui.home;

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

import com.pei.dehaze.R;
import com.pei.dehaze.databinding.FragmentHomeBinding;
import com.pei.dehaze.ui.dashboard.adapter.StatAdapter;
import com.pei.dehaze.utils.ToastUtils;

import java.util.ArrayList;
import java.util.List;

public class HomeFragment extends Fragment {

    private static final int[] STAT_IDS = new int[]{1, 2, 3, 4};

    private FragmentHomeBinding binding;
    private HomeViewModel homeViewModel;
    private StatAdapter statAdapter;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        binding = FragmentHomeBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        homeViewModel = new ViewModelProvider(this).get(HomeViewModel.class);

        initStatsRecyclerView();
        bindQuickEntries();
        bindFeatureCards();
        setupObservers();
        loadData();
    }

    private void initStatsRecyclerView() {
        statAdapter = new StatAdapter();
        binding.rvStats.setLayoutManager(new GridLayoutManager(requireContext(), 2));
        binding.rvStats.setAdapter(statAdapter);
    }

    private void bindQuickEntries() {
        // 快捷入口：处理历史 -> 我的Tab的处理历史（暂无独立Activity，跳转 profileFragment 提示）
        View historyEntry = binding.quickEntries.getChildAt(0);
        historyEntry.setOnClickListener(v -> {
            NavController navController = getNavController();
            // 处理历史入口：跳转到 profileFragment（后续 dev-personal 会细化到具体子页）
            navController.navigate(R.id.profileFragment);
        });

        // 快捷入口：我的收藏
        View favoriteEntry = binding.quickEntries.getChildAt(1);
        favoriteEntry.setOnClickListener(v -> {
            NavController navController = getNavController();
            navController.navigate(R.id.profileFragment);
        });

        // 快捷入口：批量处理 -> BatchActivity
        View batchEntry = binding.quickEntries.getChildAt(2);
        batchEntry.setOnClickListener(v -> {
            NavController navController = getNavController();
            navController.navigate(R.id.dehazeFragment);
        });

        // 快捷入口：算法选择 -> 算法列表
        View algorithmEntry = binding.quickEntries.getChildAt(3);
        algorithmEntry.setOnClickListener(v -> {
            NavController navController = getNavController();
            navController.navigate(R.id.algorithmFragment);
        });
    }

    private void bindFeatureCards() {
        // 特色能力区块在 layout 中是一个垂直 LinearLayout，包含 3 个 MaterialCardView
        // 需要找到该 LinearLayout（它是特色能力标题之后、底部间距之前的那个）
        // 特色能力区：智能算法推荐 -> 算法列表
        View featureContainer = binding.getRoot().findViewWithTag("featureCards");
        if (featureContainer instanceof ViewGroup) {
            ViewGroup fc = (ViewGroup) featureContainer;
            if (fc.getChildCount() >= 3) {
                // 智能算法推荐
                fc.getChildAt(0).setOnClickListener(v -> {
                    getNavController().navigate(R.id.algorithmFragment);
                });
                // 多模式效果对比 -> CompareActivity
                fc.getChildAt(1).setOnClickListener(v -> {
                    getNavController().navigate(R.id.action_global_compareActivity);
                });
                // 会员专属权益 -> profileFragment（后续 dev-personal 细化到会员中心）
                fc.getChildAt(2).setOnClickListener(v -> {
                    getNavController().navigate(R.id.profileFragment);
                });
            }
        }

        // CTA「开始去雾」按钮
        binding.btnStartDehaze.setOnClickListener(v -> {
            getNavController().navigate(R.id.dehazeFragment);
        });
    }

    private NavController getNavController() {
        return Navigation.findNavController(requireActivity(),
                R.id.nav_host_fragment_content_main);
    }

    private void setupObservers() {
        homeViewModel.getUserInfo().observe(getViewLifecycleOwner(), userInfo -> {
            // 用户信息展示：后续 dev-tabs 可用来个性化 Hero 区问候语
        });

        homeViewModel.getStats().observe(getViewLifecycleOwner(), this::updateStats);

        homeViewModel.getLoading().observe(getViewLifecycleOwner(), isLoading -> {
            // loading 状态暂不做全局遮罩，后续 dev-tabs 可加 shimmer 效果
        });

        homeViewModel.getError().observe(getViewLifecycleOwner(), errorMessage -> {
            if (errorMessage != null && !errorMessage.isEmpty()) {
                ToastUtils.showShort(requireContext(), errorMessage);
                homeViewModel.clearError();
            }
        });
    }

    private void loadData() {
        homeViewModel.refresh();
    }

    private void updateStats(HomeViewModel.HomeStats stats) {
        if (stats == null) return;
        List<StatAdapter.StatItem> items = new ArrayList<>();
        items.add(new StatAdapter.StatItem(STAT_IDS[0], "处理次数",
                (int) stats.getProcessCount(), "累计去雾处理次数"));
        items.add(new StatAdapter.StatItem(STAT_IDS[1], "我的收藏",
                (int) stats.getFavoriteCount(), "收藏的处理结果"));
        // 算法数量：暂无直接接口，展示为占位，后续 dev-tabs 对接算法统计接口
        items.add(new StatAdapter.StatItem(STAT_IDS[2], "可用算法",
                0, "已注册算法数量"));
        // 评分：暂无直接接口
        items.add(new StatAdapter.StatItem(STAT_IDS[3], "用户评分",
                0, "系统综合评分"));
        statAdapter.submitList(items);
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}
