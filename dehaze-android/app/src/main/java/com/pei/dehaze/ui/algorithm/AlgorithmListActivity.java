package com.pei.dehaze.ui.algorithm;

import androidx.appcompat.app.AlertDialog;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;

import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;

import com.pei.dehaze.R;
import com.pei.dehaze.databinding.ActivityAlgorithmListBinding;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmStatus;
import com.pei.dehaze.ui.algorithm.adapter.AlgorithmAdapter;
import com.pei.dehaze.ui.algorithm.viewmodel.AlgorithmViewModel;
import com.pei.dehaze.utils.StringUtils;
import com.pei.dehaze.utils.ToastUtils;

import java.util.Collections;
import java.util.List;

public class AlgorithmListActivity extends AppCompatActivity {

    private AlgorithmViewModel algorithmViewModel;
    private AlgorithmAdapter algorithmAdapter;
    private ActivityAlgorithmListBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityAlgorithmListBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        initViews();
        initViewModel();
        setupObservers();
        loadData();
    }

    private void initViews() {
        setSupportActionBar(binding.toolbar);
        binding.toolbar.setNavigationOnClickListener(v -> finish());

        algorithmAdapter = new AlgorithmAdapter();
        binding.recyclerView.setLayoutManager(new LinearLayoutManager(this));
        binding.recyclerView.setAdapter(algorithmAdapter);

        binding.swipeRefresh.setOnRefreshListener(this::loadData);

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
        });

        binding.btnSearch.setOnClickListener(v -> {
            String keywords = binding.etKeywords.getText().toString().trim();
            algorithmViewModel.setKeywords(keywords);
            loadData();
        });

        binding.btnReset.setOnClickListener(v -> {
            binding.etKeywords.setText("");
            algorithmViewModel.resetQuery();
            loadData();
        });

        binding.btnAdd.setOnClickListener(v -> showAlgorithmFormDialog(null));
    }

    private void initViewModel() {
        algorithmViewModel = new ViewModelProvider(this).get(AlgorithmViewModel.class);
    }

    private void setupObservers() {
        algorithmViewModel.getAlgorithmList().observe(this, algorithms -> {
            algorithmAdapter.setData(algorithms);
            binding.tvEmpty.setVisibility(algorithms == null || algorithms.isEmpty() ? View.VISIBLE : View.GONE);
        });

        algorithmViewModel.getLoading().observe(this, isLoading ->
                binding.swipeRefresh.setRefreshing(isLoading != null && isLoading));

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
