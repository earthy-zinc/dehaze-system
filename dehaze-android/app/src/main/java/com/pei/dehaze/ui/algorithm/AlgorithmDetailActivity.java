package com.pei.dehaze.ui.algorithm;

import androidx.appcompat.app.AlertDialog;
import android.os.Bundle;

import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.ViewModelProvider;

import com.pei.dehaze.databinding.ActivityAlgorithmDetailBinding;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmStatus;
import com.pei.dehaze.ui.algorithm.viewmodel.AlgorithmViewModel;
import com.pei.dehaze.ui.algorithm_select.viewmodel.AlgorithmSelectViewModel;
import com.pei.dehaze.utils.StringUtils;
import com.pei.dehaze.utils.ToastUtils;

import android.content.res.ColorStateList;

import java.util.Collections;
import java.util.List;

public class AlgorithmDetailActivity extends AppCompatActivity {

    private AlgorithmViewModel algorithmViewModel;
    private AlgorithmSelectViewModel selectViewModel;
    private ActivityAlgorithmDetailBinding binding;

    private long algorithmId;
    private Algorithm currentAlgorithm;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityAlgorithmDetailBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        algorithmId = getIntent().getLongExtra("algorithm_id", 0L);

        initViews();
        initViewModel();
        setupObservers();

        if (algorithmId > 0) {
            algorithmViewModel.loadAlgorithmDetail(algorithmId);
        } else {
            ToastUtils.showShort(this, "算法ID无效");
            finish();
        }
    }

    private void initViews() {
        setSupportActionBar(binding.toolbar);
        binding.toolbar.setNavigationOnClickListener(v -> finish());

        binding.btnEdit.setOnClickListener(v -> {
            if (currentAlgorithm != null) {
                showAlgorithmFormDialog(currentAlgorithm);
            }
        });

        binding.btnToggleStatus.setOnClickListener(v -> {
            if (currentAlgorithm != null) {
                showStatusTransitionDialog(currentAlgorithm);
            }
        });

        binding.btnFavorite.setOnClickListener(v -> {
            if (currentAlgorithm != null) {
                selectViewModel.addFavorite(currentAlgorithm.getId());
            }
        });

        binding.btnDelete.setOnClickListener(v -> {
            if (currentAlgorithm != null) {
                showDeleteConfirmDialog(currentAlgorithm);
            }
        });
    }

    private void initViewModel() {
        algorithmViewModel = new ViewModelProvider(this).get(AlgorithmViewModel.class);
        selectViewModel = new ViewModelProvider(this).get(AlgorithmSelectViewModel.class);
    }

    private void setupObservers() {
        algorithmViewModel.getAlgorithmDetail().observe(this, algorithm -> {
            if (algorithm != null) {
                currentAlgorithm = algorithm;
                updateUI(algorithm);
            }
        });

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
                if (algorithmId > 0) {
                    algorithmViewModel.loadAlgorithmDetail(algorithmId);
                }
            }
        });

        selectViewModel.getError().observe(this, errorMessage -> {
            if (errorMessage != null && !errorMessage.isEmpty()) {
                ToastUtils.showShort(this, errorMessage);
                selectViewModel.clearError();
            }
        });

        selectViewModel.getOperationResult().observe(this, result -> {
            if (result != null && !result.isEmpty()) {
                ToastUtils.showShort(this, result);
                selectViewModel.clearOperationResult();
            }
        });
    }

    private void updateUI(Algorithm algorithm) {
        binding.tvAlgorithmName.setText(StringUtils.safe(algorithm.getName()));
        binding.tvAlgorithmType.setText(StringUtils.safe(algorithm.getType()));
        binding.tvAlgorithmDescription.setText(StringUtils.safe(algorithm.getDescription()));
        binding.tvAlgorithmParams.setText(StringUtils.safe(algorithm.getParams()));
        binding.tvAlgorithmFlops.setText(StringUtils.safe(algorithm.getFlops()));
        binding.tvAlgorithmSize.setText(StringUtils.safe(algorithm.getSize()));
        binding.tvAlgorithmPath.setText(StringUtils.safe(algorithm.getPath()));
        binding.tvAlgorithmImportPath.setText(StringUtils.safe(algorithm.getImportPath()));

        AlgorithmStatus status = algorithm.getStatus() != null ? algorithm.getStatus() : AlgorithmStatus.DRAFT;
        binding.chipStatus.setText(status.getLabel());
        binding.chipStatus.setChipBackgroundColor(ColorStateList.valueOf(statusColor(status)));
        binding.chipStatus.setTextColor(0xFFFFFFFF);
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

    private void showDeleteConfirmDialog(Algorithm algorithm) {
        new AlertDialog.Builder(this)
                .setTitle("删除确认")
                .setMessage("确认删除算法「" + StringUtils.safe(algorithm.getName()) + "」吗？删除后不可恢复。")
                .setPositiveButton("确定", (dialog, which) -> {
                    algorithmViewModel.deleteAlgorithms(Collections.singletonList(algorithm.getId()));
                    finish();
                })
                .setNegativeButton("取消", null)
                .show();
    }

    private void showAlgorithmFormDialog(Algorithm existing) {
        AlgorithmFormDialogHelper.show(this, existing, new AlgorithmFormDialogHelper.OnSubmitListener() {
            @Override
            public void onCreate(Algorithm data) {
                // 详情页只支持编辑
            }

            @Override
            public void onUpdate(Algorithm data, long existingId) {
                algorithmViewModel.updateAlgorithm(existingId, data);
            }
        });
    }

    private int statusColor(AlgorithmStatus status) {
        if (status == null) return 0xFF9E9E9E;
        switch (status) {
            case DRAFT: return 0xFF9E9E9E;
            case TESTING: return 0xFFFF9800;
            case PENDING_AUDIT: return 0xFF2196F3;
            case PUBLISHED: return 0xFF4CAF50;
            case DISABLED: return 0xFFE53935;
            case ARCHIVED: return 0xFF607D8B;
            default: return 0xFF9E9E9E;
        }
    }

}
