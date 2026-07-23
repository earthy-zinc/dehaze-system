package com.pei.dehaze.ui.algorithm;

import android.app.AlertDialog;
import android.content.Intent;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import androidx.swiperefreshlayout.widget.SwipeRefreshLayout;

import com.google.android.material.button.MaterialButton;
import com.pei.dehaze.R;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmFavorite;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmStatus;
import com.pei.dehaze.ui.algorithm.adapter.AlgorithmAdapter;
import com.pei.dehaze.ui.algorithm.viewmodel.AlgorithmViewModel;
import com.pei.dehaze.utils.StringUtils;
import com.pei.dehaze.utils.ToastUtils;

import java.util.Collections;
import java.util.List;
import java.util.Set;

public class AlgorithmListActivity extends AppCompatActivity {

    private AlgorithmViewModel algorithmViewModel;
    private AlgorithmAdapter algorithmAdapter;
    private RecyclerView recyclerView;
    private SwipeRefreshLayout swipeRefreshLayout;
    private Toolbar toolbar;
    private TextView tvEmpty;
    private EditText etKeywords;
    private MaterialButton btnSearch;
    private MaterialButton btnReset;
    private MaterialButton btnAdd;
    private MaterialButton btnCompare;
    private MaterialButton btnCancelSelect;
    private MaterialButton btnSelectAll;
    private MaterialButton btnFavorites;
    private TextView tvSelectionInfo;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_algorithm_list);

        initViews();
        initViewModel();
        setupObservers();
        loadData();
    }

    private void initViews() {
        toolbar = findViewById(R.id.toolbar);
        recyclerView = findViewById(R.id.recycler_view);
        swipeRefreshLayout = findViewById(R.id.swipe_refresh);
        tvEmpty = findViewById(R.id.tv_empty);
        etKeywords = findViewById(R.id.et_keywords);
        btnSearch = findViewById(R.id.btn_search);
        btnReset = findViewById(R.id.btn_reset);
        btnAdd = findViewById(R.id.btn_add);
        btnCompare = findViewById(R.id.btn_compare);
        btnCancelSelect = findViewById(R.id.btn_cancel_select);
        btnSelectAll = findViewById(R.id.btn_select_all);
        btnFavorites = findViewById(R.id.btn_favorites);
        tvSelectionInfo = findViewById(R.id.tv_selection_info);

        setSupportActionBar(toolbar);
        toolbar.setNavigationOnClickListener(v -> finish());

        algorithmAdapter = new AlgorithmAdapter();
        recyclerView.setLayoutManager(new LinearLayoutManager(this));
        recyclerView.setAdapter(algorithmAdapter);

        swipeRefreshLayout.setOnRefreshListener(this::loadData);

        algorithmAdapter.setOnAlgorithmActionListener(new AlgorithmAdapter.OnAlgorithmActionListener() {
            @Override
            public void onView(Algorithm algorithm) {
                Intent intent = new Intent(AlgorithmListActivity.this, AlgorithmDetailActivity.class);
                intent.putExtra("algorithm_id", algorithm.getId());
                startActivity(intent);
            }

            @Override
            public void onEdit(Algorithm algorithm) {
                showAlgorithmFormDialog(algorithm);
            }

            @Override
            public void onDelete(Algorithm algorithm) {
                showDeleteConfirmDialog(algorithm);
            }

            @Override
            public void onToggleStatus(Algorithm algorithm) {
                showStatusTransitionDialog(algorithm);
            }

            @Override
            public void onToggleFavorite(Algorithm algorithm) {
                algorithmViewModel.toggleFavorite(algorithm.getId());
            }
        });

        algorithmAdapter.setOnSelectionChangedListener(selectedIds -> {
            tvSelectionInfo.setText("已选中 " + selectedIds.size() + " 个算法");
        });

        btnSearch.setOnClickListener(v -> {
            String keywords = etKeywords.getText().toString().trim();
            algorithmViewModel.setKeywords(keywords);
            loadData();
        });

        btnReset.setOnClickListener(v -> {
            etKeywords.setText("");
            algorithmViewModel.resetQuery();
            loadData();
        });

        btnAdd.setOnClickListener(v -> showAlgorithmFormDialog(null));

        btnCompare.setOnClickListener(v -> {
            if (!algorithmAdapter.isSelectionMode()) {
                algorithmAdapter.setSelectionMode(true);
                updateSelectionModeUI(true);
                ToastUtils.showShort(this, "长按或勾选要对比的算法（至少2个）");
            } else {
                showCompareConfirmDialog();
            }
        });

        btnCancelSelect.setOnClickListener(v -> {
            algorithmAdapter.clearSelection();
            algorithmAdapter.setSelectionMode(false);
            updateSelectionModeUI(false);
        });

        btnSelectAll.setOnClickListener(v -> algorithmAdapter.selectAll());

        btnFavorites.setOnClickListener(v -> algorithmViewModel.loadFavorites());
    }

    private void updateSelectionModeUI(boolean selectionMode) {
        btnCancelSelect.setVisibility(selectionMode ? View.VISIBLE : View.GONE);
        btnSelectAll.setVisibility(selectionMode ? View.VISIBLE : View.GONE);
        tvSelectionInfo.setVisibility(selectionMode ? View.VISIBLE : View.GONE);
        btnAdd.setVisibility(selectionMode ? View.GONE : View.VISIBLE);
        btnFavorites.setVisibility(selectionMode ? View.GONE : View.VISIBLE);
        btnCompare.setText(selectionMode ? "开始对比" : "对比");
    }

    private void initViewModel() {
        algorithmViewModel = new ViewModelProvider(this).get(AlgorithmViewModel.class);
    }

    private void setupObservers() {
        algorithmViewModel.getAlgorithmList().observe(this, algorithms -> {
            algorithmAdapter.setData(algorithms);
            tvEmpty.setVisibility(algorithms == null || algorithms.isEmpty() ? View.VISIBLE : View.GONE);
        });

        algorithmViewModel.getLoading().observe(this, isLoading ->
                swipeRefreshLayout.setRefreshing(isLoading != null && isLoading));

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
                if (result.startsWith("删除") || result.startsWith("状态")) {
                    if (algorithmAdapter.isSelectionMode()) {
                        algorithmAdapter.setSelectionMode(false);
                        updateSelectionModeUI(false);
                    }
                }
            }
        });

        algorithmViewModel.getCompareResult().observe(this, compareList -> {
            if (compareList != null && !compareList.isEmpty()) {
                showCompareResultDialog(compareList);
                algorithmViewModel.clearCompareResult();
            }
        });

        algorithmViewModel.getFavoriteList().observe(this, favorites -> {
            if (favorites != null) {
                showFavoritesDialog(favorites);
            }
        });
    }

    private void loadData() {
        algorithmViewModel.loadAlgorithms();
    }

    private void showDeleteConfirmDialog(Algorithm algorithm) {
        new AlertDialog.Builder(this)
                .setTitle("删除确认")
                .setMessage("确认删除算法「" + StringUtils.safe(algorithm.getName()) + "」吗？删除后不可恢复。")
                .setPositiveButton("确定", (dialog, which) ->
                        algorithmViewModel.deleteAlgorithms(Collections.singletonList(algorithm.getId())))
                .setNegativeButton("取消", null)
                .show();
    }

    private void showStatusTransitionDialog(Algorithm algorithm) {
        AlgorithmStatus currentStatus = algorithm.getStatus() != null ? algorithm.getStatus() : AlgorithmStatus.DRAFT;
        List<AlgorithmStatus> nextStatuses = currentStatus.nextStatuses();
        if (nextStatuses.isEmpty()) {
            ToastUtils.showShort(this, "当前状态「" + currentStatus.getLabel() + "」不可流转");
            return;
        }
        String[] items = new String[nextStatuses.size()];
        for (int i = 0; i < nextStatuses.size(); i++) {
            items[i] = nextStatuses.get(i).getLabel();
        }
        new AlertDialog.Builder(this)
                .setTitle("状态流转 - " + StringUtils.safe(algorithm.getName()))
                .setItems(items, (dialog, which) -> {
                    AlgorithmStatus newStatus = nextStatuses.get(which);
                    algorithmViewModel.updateAlgorithmStatus(algorithm.getId(), newStatus);
                })
                .show();
    }

    private void showCompareConfirmDialog() {
        Set<Long> selectedIds = algorithmAdapter.getSelectedIds();
        if (selectedIds.size() < 2) {
            ToastUtils.showShort(this, "对比至少需要选择2个算法");
            return;
        }
        new AlertDialog.Builder(this)
                .setTitle("算法对比")
                .setMessage("确认对比选中的 " + selectedIds.size() + " 个算法吗？")
                .setPositiveButton("确定", (dialog, which) -> {
                    String ids = algorithmAdapter.getSelectedIdsString();
                    algorithmViewModel.compareAlgorithms(ids);
                })
                .setNegativeButton("取消", null)
                .show();
    }

    private void showCompareResultDialog(List<Algorithm> compareList) {
        View view = LayoutInflater.from(this).inflate(R.layout.dialog_algorithm_compare, null);
        LinearLayout container = view.findViewById(R.id.container_compare_items);
        TextView tvSummary = view.findViewById(R.id.tv_compare_summary);

        for (Algorithm algorithm : compareList) {
            View itemView = LayoutInflater.from(this).inflate(R.layout.item_algorithm_compare, container, false);
            ((TextView) itemView.findViewById(R.id.tv_name)).setText(StringUtils.safe(algorithm.getName()));
            ((TextView) itemView.findViewById(R.id.tv_type)).setText("类型: " + StringUtils.safe(algorithm.getType()));
            ((TextView) itemView.findViewById(R.id.tv_params)).setText("参数量: " + StringUtils.safe(algorithm.getParams()));
            ((TextView) itemView.findViewById(R.id.tv_flops)).setText("FLOPs: " + StringUtils.safe(algorithm.getFlops()));
            AlgorithmStatus status = algorithm.getStatus();
            ((TextView) itemView.findViewById(R.id.tv_status)).setText("状态: " + (status != null ? status.getLabel() : ""));
            ((TextView) itemView.findViewById(R.id.tv_description)).setText(StringUtils.safe(algorithm.getDescription()));
            container.addView(itemView);
        }

        StringBuilder summary = new StringBuilder("共对比 " + compareList.size() + " 个算法\n");
        for (Algorithm algorithm : compareList) {
            summary.append("• ").append(StringUtils.safe(algorithm.getName()))
                    .append("（参数: ").append(StringUtils.safe(algorithm.getParams()))
                    .append(", FLOPs: ").append(StringUtils.safe(algorithm.getFlops()))
                    .append("）\n");
        }
        tvSummary.setText(summary.toString());

        new AlertDialog.Builder(this)
                .setTitle("算法对比结果")
                .setView(view)
                .setPositiveButton("关闭", null)
                .show();

        algorithmAdapter.setSelectionMode(false);
        updateSelectionModeUI(false);
    }

    private void showFavoritesDialog(List<AlgorithmFavorite> favorites) {
        if (favorites.isEmpty()) {
            ToastUtils.showShort(this, "暂无收藏的算法");
            return;
        }
        String[] items = new String[favorites.size()];
        for (int i = 0; i < favorites.size(); i++) {
            AlgorithmFavorite fav = favorites.get(i);
            items[i] = "算法ID: " + fav.getAlgorithmId() + "（收藏时间: " + StringUtils.safe(fav.getCreateTime()) + "）";
        }
        new AlertDialog.Builder(this)
                .setTitle("收藏夹（" + favorites.size() + "）")
                .setItems(items, (dialog, which) -> {
                    AlgorithmFavorite fav = favorites.get(which);
                    Intent intent = new Intent(this, AlgorithmDetailActivity.class);
                    Long algId = fav.getAlgorithmId();
                    intent.putExtra("algorithm_id", algId != null ? algId : 0L);
                    startActivity(intent);
                })
                .setPositiveButton("关闭", null)
                .show();
    }

    private void showAlgorithmFormDialog(Algorithm existing) {
        AlgorithmFormDialogHelper.show(this, existing, new AlgorithmFormDialogHelper.OnSubmitListener() {
            @Override
            public void onCreate(Algorithm data) {
                algorithmViewModel.addAlgorithm(data);
            }

            @Override
            public void onUpdate(Algorithm data, long existingId) {
                algorithmViewModel.updateAlgorithm(existingId, data);
            }
        });
    }

}
