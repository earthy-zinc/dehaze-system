package com.pei.dehaze.ui.algorithm_select;

import android.os.Bundle;
import android.text.TextUtils;
import android.widget.ArrayAdapter;
import android.widget.Spinner;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import androidx.swiperefreshlayout.widget.SwipeRefreshLayout;

import com.google.android.material.button.MaterialButton;
import com.google.android.material.tabs.TabLayout;
import com.google.android.material.textfield.TextInputEditText;
import com.pei.dehaze.R;
import com.pei.dehaze.sdk.model.algorithm_select.AlgorithmCompareVO;
import com.pei.dehaze.sdk.model.algorithm_select.AlgorithmRecommendVO;
import com.pei.dehaze.sdk.model.algorithm_select.FavoriteVO;
import com.pei.dehaze.ui.algorithm_select.adapter.AlgorithmCompareResultAdapter;
import com.pei.dehaze.ui.algorithm_select.adapter.AlgorithmFavoriteAdapter;
import com.pei.dehaze.ui.algorithm_select.adapter.AlgorithmRecommendAdapter;
import com.pei.dehaze.ui.algorithm_select.viewmodel.AlgorithmSelectViewModel;
import com.pei.dehaze.utils.ToastUtils;

import android.app.AlertDialog;

import java.util.ArrayList;
import java.util.List;

public class AlgorithmSelectActivity extends AppCompatActivity {

    private static final Integer[] TOP_N_VALUES = {1, 2, 3, 5, 10};

    private AlgorithmSelectViewModel viewModel;
    private Toolbar toolbar;
    private TabLayout tabLayout;
    private android.widget.ViewFlipper viewFlipper;

    // 推荐 Tab
    private TextInputEditText etRecommendImageUrl;
    private Spinner spinnerTopN;
    private MaterialButton btnRecommend;
    private SwipeRefreshLayout swipeRecommend;
    private RecyclerView rvRecommend;
    private AlgorithmRecommendAdapter recommendAdapter;

    // 收藏 Tab
    private MaterialButton btnRefreshFavorites;
    private SwipeRefreshLayout swipeFavorites;
    private RecyclerView rvFavorites;
    private AlgorithmFavoriteAdapter favoriteAdapter;

    // 对比 Tab
    private TextInputEditText etCompareImageUrl;
    private TextInputEditText etAlgorithmIds;
    private MaterialButton btnCompare;
    private SwipeRefreshLayout swipeCompare;
    private RecyclerView rvCompare;
    private AlgorithmCompareResultAdapter compareAdapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_algorithm_select);

        initViews();
        initViewModel();
        setupObservers();
        loadFavorites();
    }

    private void initViews() {
        toolbar = findViewById(R.id.toolbar);
        tabLayout = findViewById(R.id.tab_layout);
        viewFlipper = findViewById(R.id.view_flipper);

        setSupportActionBar(toolbar);
        toolbar.setNavigationOnClickListener(v -> finish());

        tabLayout.addTab(tabLayout.newTab().setText("智能推荐"));
        tabLayout.addTab(tabLayout.newTab().setText("我的收藏"));
        tabLayout.addTab(tabLayout.newTab().setText("算法对比"));
        tabLayout.addOnTabSelectedListener(new TabLayout.OnTabSelectedListener() {
            @Override
            public void onTabSelected(TabLayout.Tab tab) {
                viewFlipper.setDisplayedChild(tab.getPosition());
                if (tab.getPosition() == 1) {
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

        // 推荐 Tab
        etRecommendImageUrl = findViewById(R.id.et_recommend_image_url);
        spinnerTopN = findViewById(R.id.spinner_top_n);
        btnRecommend = findViewById(R.id.btn_recommend);
        swipeRecommend = findViewById(R.id.swipe_recommend);
        rvRecommend = findViewById(R.id.rv_recommend);

        String[] topNLabels = {"1", "2", "3", "5", "10"};
        ArrayAdapter<String> topNAdapter = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_item, topNLabels);
        topNAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinnerTopN.setAdapter(topNAdapter);
        spinnerTopN.setSelection(2); // 默认3

        recommendAdapter = new AlgorithmRecommendAdapter();
        rvRecommend.setLayoutManager(new LinearLayoutManager(this));
        rvRecommend.setAdapter(recommendAdapter);

        swipeRecommend.setOnRefreshListener(this::doRecommend);

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

        btnRecommend.setOnClickListener(v -> doRecommend());

        // 收藏 Tab
        btnRefreshFavorites = findViewById(R.id.btn_refresh_favorites);
        swipeFavorites = findViewById(R.id.swipe_favorites);
        rvFavorites = findViewById(R.id.rv_favorites);

        favoriteAdapter = new AlgorithmFavoriteAdapter();
        rvFavorites.setLayoutManager(new LinearLayoutManager(this));
        rvFavorites.setAdapter(favoriteAdapter);

        swipeFavorites.setOnRefreshListener(this::loadFavorites);
        btnRefreshFavorites.setOnClickListener(v -> loadFavorites());

        favoriteAdapter.setOnFavoriteActionListener(new AlgorithmFavoriteAdapter.OnFavoriteActionListener() {
            @Override
            public void onUse(FavoriteVO vo) {
                ToastUtils.showShort(AlgorithmSelectActivity.this,
                        "已选择算法：" + vo.getAlgorithmName());
            }

            @Override
            public void onCancelFavorite(FavoriteVO vo) {
                showCancelFavoriteConfirmDialog(vo);
            }
        });

        // 对比 Tab
        etCompareImageUrl = findViewById(R.id.et_compare_image_url);
        etAlgorithmIds = findViewById(R.id.et_algorithm_ids);
        btnCompare = findViewById(R.id.btn_compare);
        swipeCompare = findViewById(R.id.swipe_compare);
        rvCompare = findViewById(R.id.rv_compare);

        compareAdapter = new AlgorithmCompareResultAdapter();
        rvCompare.setLayoutManager(new LinearLayoutManager(this,
                LinearLayoutManager.HORIZONTAL, false));
        rvCompare.setAdapter(compareAdapter);

        btnCompare.setOnClickListener(v -> doCompare());
    }

    private void initViewModel() {
        viewModel = new ViewModelProvider(this).get(AlgorithmSelectViewModel.class);
    }

    private void setupObservers() {
        viewModel.getRecommendList().observe(this, list ->
                recommendAdapter.submitList(list));

        viewModel.getFavoriteList().observe(this, list ->
                favoriteAdapter.submitList(list));

        viewModel.getCompareResult().observe(this, list ->
                compareAdapter.submitList(list));

        viewModel.getLoading().observe(this, isLoading -> {
            boolean loading = isLoading != null && isLoading;
            swipeRecommend.setRefreshing(loading);
            swipeFavorites.setRefreshing(loading);
            swipeCompare.setRefreshing(loading);
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
    }

    private void doRecommend() {
        String imageUrl = etRecommendImageUrl.getText() == null ? "" : etRecommendImageUrl.getText().toString().trim();
        if (TextUtils.isEmpty(imageUrl)) {
            ToastUtils.showShort(this, "请输入待去雾图片URL");
            return;
        }
        int topN = TOP_N_VALUES[spinnerTopN.getSelectedItemPosition()];
        viewModel.recommend(imageUrl, topN);
    }

    private void loadFavorites() {
        viewModel.loadFavorites();
    }

    private void doCompare() {
        String imageUrl = etCompareImageUrl.getText() == null ? "" : etCompareImageUrl.getText().toString().trim();
        String idsStr = etAlgorithmIds.getText() == null ? "" : etAlgorithmIds.getText().toString().trim();
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
