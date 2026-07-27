package com.pei.dehaze.ui.algorithm_select;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.View;
import android.view.inputmethod.EditorInfo;
import android.widget.ArrayAdapter;

import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;

import com.google.android.material.tabs.TabLayout;
import com.pei.dehaze.R;
import com.pei.dehaze.databinding.ActivityAlgorithmSelectBinding;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.sdk.model.algorithm_select.AlgorithmCompareVO;
import com.pei.dehaze.sdk.model.algorithm_select.AlgorithmRecommendVO;
import com.pei.dehaze.sdk.model.algorithm_select.FavoriteVO;
import com.pei.dehaze.ui.algorithm.viewmodel.AlgorithmViewModel;
import com.pei.dehaze.ui.algorithm_select.adapter.AlgorithmBrowseAdapter;
import com.pei.dehaze.ui.algorithm_select.adapter.AlgorithmCompareResultAdapter;
import com.pei.dehaze.ui.algorithm_select.adapter.AlgorithmFavoriteAdapter;
import com.pei.dehaze.ui.algorithm_select.adapter.AlgorithmRecommendAdapter;
import com.pei.dehaze.ui.algorithm_select.viewmodel.AlgorithmSelectViewModel;
import com.pei.dehaze.utils.ToastUtils;

import androidx.appcompat.app.AlertDialog;

import java.util.ArrayList;
import java.util.List;

public class AlgorithmSelectActivity extends AppCompatActivity {

    public static final String EXTRA_ALGORITHM_ID = "extra_algorithm_id";
    public static final String EXTRA_ALGORITHM_NAME = "extra_algorithm_name";
    private static final Integer[] TOP_N_VALUES = {1, 2, 3, 5, 10};
    private static final int TAB_SEARCH = 0;
    private static final int TAB_RECOMMEND = 1;
    private static final int TAB_FAVORITES = 2;
    private static final int TAB_COMPARE = 3;

    private AlgorithmSelectViewModel viewModel;
    private AlgorithmViewModel algorithmViewModel;
    private ActivityAlgorithmSelectBinding binding;

    // 搜索 Tab
    private AlgorithmBrowseAdapter browseAdapter;

    // 推荐 Tab
    private AlgorithmRecommendAdapter recommendAdapter;

    // 收藏 Tab
    private AlgorithmFavoriteAdapter favoriteAdapter;

    // 对比 Tab
    private AlgorithmCompareResultAdapter compareAdapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityAlgorithmSelectBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        initViews();
        initViewModel();
        setupObservers();
        loadFavorites();
        algorithmViewModel.loadAlgorithms();
    }

    private void initViews() {
        setSupportActionBar(binding.toolbar);
        binding.toolbar.setNavigationOnClickListener(v -> finish());

        binding.tabLayout.addTab(binding.tabLayout.newTab().setText("搜索"));
        binding.tabLayout.addTab(binding.tabLayout.newTab().setText("智能推荐"));
        binding.tabLayout.addTab(binding.tabLayout.newTab().setText("我的收藏"));
        binding.tabLayout.addTab(binding.tabLayout.newTab().setText("算法对比"));
        binding.tabLayout.addOnTabSelectedListener(new TabLayout.OnTabSelectedListener() {
            @Override
            public void onTabSelected(TabLayout.Tab tab) {
                binding.viewFlipper.setDisplayedChild(tab.getPosition());
                if (tab.getPosition() == TAB_FAVORITES) {
                    loadFavorites();
                }
            }

            @Override
            public void onTabUnselected(TabLayout.Tab tab) {
            }

            @Override
            public void onTabReselected(TabLayout.Tab tab) {
            }
        });

        // 搜索 Tab
        browseAdapter = new AlgorithmBrowseAdapter();
        binding.rvSearch.setLayoutManager(new LinearLayoutManager(this));
        binding.rvSearch.setAdapter(browseAdapter);

        binding.swipeSearch.setOnRefreshListener(() -> algorithmViewModel.loadAlgorithms());
        binding.btnSearchAlgo.setOnClickListener(v -> doSearch());
        binding.btnResetSearch.setOnClickListener(v -> {
            binding.etSearchKeywords.setText("");
            algorithmViewModel.resetQuery();
            algorithmViewModel.loadAlgorithms();
        });
        binding.etSearchKeywords.setOnEditorActionListener((v, actionId, event) -> {
            if (actionId == EditorInfo.IME_ACTION_SEARCH) {
                doSearch();
                return true;
            }
            return false;
        });

        browseAdapter.setOnBrowseActionListener(new AlgorithmBrowseAdapter.OnBrowseActionListener() {
            @Override
            public void onUse(Algorithm algorithm) {
                Intent data = new Intent();
                data.putExtra(EXTRA_ALGORITHM_ID, algorithm.getId());
                data.putExtra(EXTRA_ALGORITHM_NAME, algorithm.getName());
                setResult(Activity.RESULT_OK, data);
                ToastUtils.showShort(AlgorithmSelectActivity.this,
                        "已选择算法：" + algorithm.getName());
                finish();
            }

            @Override
            public void onFavorite(Algorithm algorithm) {
                viewModel.toggleFavorite(algorithm.getId());
            }
        });

        // 推荐 Tab
        String[] topNLabels = {"1", "2", "3", "5", "10"};
        ArrayAdapter<String> topNAdapter = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_item, topNLabels);
        topNAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        binding.spinnerTopN.setAdapter(topNAdapter);
        binding.spinnerTopN.setSelection(2); // 默认3

        recommendAdapter = new AlgorithmRecommendAdapter();
        binding.rvRecommend.setLayoutManager(new LinearLayoutManager(this));
        binding.rvRecommend.setAdapter(recommendAdapter);

        binding.swipeRecommend.setOnRefreshListener(this::doRecommend);

        recommendAdapter.setOnRecommendActionListener(new AlgorithmRecommendAdapter.OnRecommendActionListener() {
            @Override
            public void onUse(AlgorithmRecommendVO vo) {
                ToastUtils.showShort(AlgorithmSelectActivity.this,
                        "已选择算法：" + vo.getAlgorithmName());
            }

            @Override
            public void onFavorite(AlgorithmRecommendVO vo) {
                viewModel.toggleFavorite(vo.getAlgorithmId());
            }
        });

        binding.btnRecommend.setOnClickListener(v -> doRecommend());

        // 收藏 Tab
        favoriteAdapter = new AlgorithmFavoriteAdapter();
        binding.rvFavorites.setLayoutManager(new LinearLayoutManager(this));
        binding.rvFavorites.setAdapter(favoriteAdapter);

        binding.swipeFavorites.setOnRefreshListener(this::loadFavorites);
        binding.btnRefreshFavorites.setOnClickListener(v -> loadFavorites());

        favoriteAdapter.setOnFavoriteActionListener(new AlgorithmFavoriteAdapter.OnFavoriteActionListener() {
            @Override
            public void onUse(FavoriteVO vo) {
                Intent data = new Intent();
                data.putExtra(EXTRA_ALGORITHM_ID, vo.getAlgorithmId());
                data.putExtra(EXTRA_ALGORITHM_NAME, vo.getAlgorithmName());
                setResult(Activity.RESULT_OK, data);
                finish();
            }

            @Override
            public void onCancelFavorite(FavoriteVO vo) {
                showCancelFavoriteConfirmDialog(vo);
            }
        });

        // 对比 Tab
        compareAdapter = new AlgorithmCompareResultAdapter();
        binding.rvCompare.setLayoutManager(new LinearLayoutManager(this,
                LinearLayoutManager.HORIZONTAL, false));
        binding.rvCompare.setAdapter(compareAdapter);

        binding.btnCompare.setOnClickListener(v -> doCompare());
    }

    private void doSearch() {
        String keywords = binding.etSearchKeywords.getText() == null ? "" : binding.etSearchKeywords.getText().toString().trim();
        algorithmViewModel.setKeywords(keywords);
        algorithmViewModel.loadAlgorithms();
    }

    private void initViewModel() {
        viewModel = new ViewModelProvider(this).get(AlgorithmSelectViewModel.class);
        algorithmViewModel = new ViewModelProvider(this).get(AlgorithmViewModel.class);
    }

    private void setupObservers() {
        viewModel.getRecommendList().observe(this, list -> {
            recommendAdapter.submitList(list);
            binding.tvEmptyRecommend.setVisibility(list == null || list.isEmpty() ? View.VISIBLE : View.GONE);
        });

        viewModel.getFavoriteList().observe(this, list -> {
            favoriteAdapter.submitList(list);
            binding.tvEmptyFavorites.setVisibility(list == null || list.isEmpty() ? View.VISIBLE : View.GONE);
        });

        viewModel.getCompareResult().observe(this, list -> {
            compareAdapter.submitList(list);
            binding.tvEmptyCompare.setVisibility(list == null || list.isEmpty() ? View.VISIBLE : View.GONE);
        });

        viewModel.getLoading().observe(this, isLoading -> {
            boolean loading = isLoading != null && isLoading;
            binding.swipeRecommend.setRefreshing(loading);
            binding.swipeFavorites.setRefreshing(loading);
            binding.swipeCompare.setRefreshing(loading);
        });

        viewModel.getError().observe(this, errorMessage -> {
            if (errorMessage != null && !errorMessage.isEmpty()) {
                ToastUtils.showShort(this, errorMessage);
                viewModel.clearError();
            }
        });

        viewModel.getOperationResult().observe(this, result -> {
            if (result != null && !result.isEmpty()) {
                ToastUtils.showShort(this, result);
                viewModel.clearOperationResult();
                if (result.contains("收藏")) {
                    loadFavorites();
                }
            }
        });

        viewModel.getFavoriteToggleResult().observe(this, result -> {
            if (result != null) {
                viewModel.clearFavoriteToggleResult();
            }
        });

        // 搜索 Tab 的观察者
        algorithmViewModel.getAlgorithmList().observe(this, list -> {
            browseAdapter.submitList(list);
            binding.tvEmptySearch.setVisibility(list == null || list.isEmpty() ? View.VISIBLE : View.GONE);
        });

        algorithmViewModel.getLoading().observe(this, isLoading ->
                binding.swipeSearch.setRefreshing(isLoading != null && isLoading));

        algorithmViewModel.getError().observe(this, errorMessage -> {
            if (errorMessage != null && !errorMessage.isEmpty()) {
                ToastUtils.showShort(this, errorMessage);
                algorithmViewModel.clearError();
            }
        });

        algorithmViewModel.getOperationResult().observe(this, result -> {
            if (result != null && !result.isEmpty()) {
                ToastUtils.showShort(this, result);
                algorithmViewModel.clearOperationResult();
            }
        });
    }

    private void doRecommend() {
        String imageUrl = binding.etRecommendImageUrl.getText() == null ? "" : binding.etRecommendImageUrl.getText().toString().trim();
        if (TextUtils.isEmpty(imageUrl)) {
            ToastUtils.showShort(this, "请输入待去雾图片URL");
            return;
        }
        int topN = TOP_N_VALUES[binding.spinnerTopN.getSelectedItemPosition()];
        viewModel.recommend(imageUrl, topN);
    }

    private void loadFavorites() {
        viewModel.loadFavorites();
    }

    private void doCompare() {
        String imageUrl = binding.etCompareImageUrl.getText() == null ? "" : binding.etCompareImageUrl.getText().toString().trim();
        String idsStr = binding.etAlgorithmIds.getText() == null ? "" : binding.etAlgorithmIds.getText().toString().trim();
        if (TextUtils.isEmpty(idsStr)) {
            ToastUtils.showShort(this, "请输入算法ID列表");
            return;
        }
        String[] parts = idsStr.split(",");
        if (parts.length < 2 || parts.length > 4) {
            ToastUtils.showShort(this, "请输入2-4个算法ID");
            return;
        }
        List<Long> algorithmIds = new ArrayList<>();
        for (String part : parts) {
            try {
                algorithmIds.add(Long.parseLong(part.trim()));
            } catch (NumberFormatException e) {
                ToastUtils.showShort(this, "算法ID格式错误：" + part);
                return;
            }
        }
        viewModel.compare(algorithmIds, imageUrl);
    }

    private void showCancelFavoriteConfirmDialog(FavoriteVO vo) {
        new AlertDialog.Builder(this)
                .setTitle("取消收藏")
                .setMessage("确认取消收藏「" + (vo.getAlgorithmName() == null ? "" : vo.getAlgorithmName()) + "」吗？")
                .setPositiveButton("确定", (dialog, which) ->
                        viewModel.toggleFavorite(vo.getAlgorithmId()))
                .setNegativeButton("取消", null)
                .show();
    }
}
