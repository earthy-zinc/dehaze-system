package com.pei.dehaze.ui.algorithm_select;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.inputmethod.EditorInfo;
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

import android.app.AlertDialog;

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
    private Toolbar toolbar;
    private TabLayout tabLayout;
    private android.widget.ViewFlipper viewFlipper;

    // 搜索 Tab
    private TextInputEditText etSearchKeywords;
    private MaterialButton btnSearchAlgo;
    private MaterialButton btnResetSearch;
    private SwipeRefreshLayout swipeSearch;
    private RecyclerView rvSearch;
    private AlgorithmBrowseAdapter browseAdapter;

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
        algorithmViewModel.loadAlgorithms();
    }

    private void initViews() {
        toolbar = findViewById(R.id.toolbar);
        tabLayout = findViewById(R.id.tab_layout);
        viewFlipper = findViewById(R.id.view_flipper);

        setSupportActionBar(toolbar);
        toolbar.setNavigationOnClickListener(v -> finish());

        tabLayout.addTab(tabLayout.newTab().setText("全部算法"));
        tabLayout.addTab(tabLayout.newTab().setText("智能推荐"));
        tabLayout.addTab(tabLayout.newTab().setText("我的收藏"));
        tabLayout.addTab(tabLayout.newTab().setText("算法对比"));
        tabLayout.addOnTabSelectedListener(new TabLayout.OnTabSelectedListener() {
            @Override
            public void onTabSelected(TabLayout.Tab tab) {
                viewFlipper.setDisplayedChild(tab.getPosition());
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
        etSearchKeywords = findViewById(R.id.et_search_keywords);
        btnSearchAlgo = findViewById(R.id.btn_search_algo);
        btnResetSearch = findViewById(R.id.btn_reset_search);
        swipeSearch = findViewById(R.id.swipe_search);
        rvSearch = findViewById(R.id.rv_search);

        browseAdapter = new AlgorithmBrowseAdapter();
        rvSearch.setLayoutManager(new LinearLayoutManager(this));
        rvSearch.setAdapter(browseAdapter);

        swipeSearch.setOnRefreshListener(() -> algorithmViewModel.loadAlgorithms());
        btnSearchAlgo.setOnClickListener(v -> doSearch());
        btnResetSearch.setOnClickListener(v -> {
            etSearchKeywords.setText("");
            algorithmViewModel.resetQuery();
            algorithmViewModel.loadAlgorithms();
        });
        etSearchKeywords.setOnEditorActionListener((v, actionId, event) -> {
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
                algorithmViewModel.toggleFavorite(algorithm.getId());
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

    private void doSearch() {
        String keywords = etSearchKeywords.getText() == null ? "" : etSearchKeywords.getText().toString().trim();
        algorithmViewModel.setKeywords(keywords);
        algorithmViewModel.loadAlgorithms();
    }

    private void initViewModel() {
        viewModel = new ViewModelProvider(this).get(AlgorithmSelectViewModel.class);
        algorithmViewModel = new ViewModelProvider(this).get(AlgorithmViewModel.class);
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

        // 搜索 Tab 的观察者
        algorithmViewModel.getAlgorithmList().observe(this, list ->
                browseAdapter.submitList(list));

        algorithmViewModel.getLoading().observe(this, isLoading ->
                swipeSearch.setRefreshing(isLoading != null && isLoading));

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
