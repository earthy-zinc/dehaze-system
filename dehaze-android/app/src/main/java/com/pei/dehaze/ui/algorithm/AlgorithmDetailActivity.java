package com.pei.dehaze.ui.algorithm;

import android.app.AlertDialog;
import android.os.Bundle;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.lifecycle.ViewModelProvider;

import com.google.android.material.button.MaterialButton;
import com.google.android.material.chip.Chip;
import com.pei.dehaze.R;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmStatus;
import com.pei.dehaze.ui.algorithm.viewmodel.AlgorithmViewModel;
import com.pei.dehaze.utils.StringUtils;
import com.pei.dehaze.utils.ToastUtils;

import android.content.res.ColorStateList;

import java.util.Collections;
import java.util.List;

public class AlgorithmDetailActivity extends AppCompatActivity {

    private AlgorithmViewModel algorithmViewModel;
    private Toolbar toolbar;
    private TextView tvName, tvType, tvDescription, tvParams, tvFlops, tvSize, tvPath, tvImportPath;
    private Chip chipStatus;
    private MaterialButton btnEdit, btnToggleStatus, btnFavorite, btnDelete;

    private long algorithmId;
    private Algorithm currentAlgorithm;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_algorithm_detail);

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
        toolbar = findViewById(R.id.toolbar);
        tvName = findViewById(R.id.tv_algorithm_name);
        tvType = findViewById(R.id.tv_algorithm_type);
        tvDescription = findViewById(R.id.tv_algorithm_description);
        tvParams = findViewById(R.id.tv_algorithm_params);
        tvFlops = findViewById(R.id.tv_algorithm_flops);
        tvSize = findViewById(R.id.tv_algorithm_size);
        tvPath = findViewById(R.id.tv_algorithm_path);
        tvImportPath = findViewById(R.id.tv_algorithm_import_path);
        chipStatus = findViewById(R.id.chip_status);
        btnEdit = findViewById(R.id.btn_edit);
        btnToggleStatus = findViewById(R.id.btn_toggle_status);
        btnFavorite = findViewById(R.id.btn_favorite);
        btnDelete = findViewById(R.id.btn_delete);

        setSupportActionBar(toolbar);
        toolbar.setNavigationOnClickListener(v -> finish());

        btnEdit.setOnClickListener(v -> {
            if (currentAlgorithm != null) {
                showAlgorithmFormDialog(currentAlgorithm);
            }
        });

        btnToggleStatus.setOnClickListener(v -> {
            if (currentAlgorithm != null) {
                showStatusTransitionDialog(currentAlgorithm);
            }
        });

        btnFavorite.setOnClickListener(v -> {
            if (currentAlgorithm != null) {
                algorithmViewModel.toggleFavorite(currentAlgorithm.getId());
            }
        });

        btnDelete.setOnClickListener(v -> {
            if (currentAlgorithm != null) {
                showDeleteConfirmDialog(currentAlgorithm);
            }
        });
    }

    private void initViewModel() {
        algorithmViewModel = new ViewModelProvider(this).get(AlgorithmViewModel.class);
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
    }

    private void updateUI(Algorithm algorithm) {
        tvName.setText(StringUtils.safe(algorithm.getName()));
        tvType.setText(StringUtils.safe(algorithm.getType()));
        tvDescription.setText(StringUtils.safe(algorithm.getDescription()));
        tvParams.setText(StringUtils.safe(algorithm.getParams()));
        tvFlops.setText(StringUtils.safe(algorithm.getFlops()));
        tvSize.setText(StringUtils.safe(algorithm.getSize()));
        tvPath.setText(StringUtils.safe(algorithm.getPath()));
        tvImportPath.setText(StringUtils.safe(algorithm.getImportPath()));

        AlgorithmStatus status = algorithm.getStatus() != null ? algorithm.getStatus() : AlgorithmStatus.DRAFT;
        chipStatus.setText(status.getLabel());
        chipStatus.setChipBackgroundColor(ColorStateList.valueOf(statusColor(status)));
        chipStatus.setTextColor(0xFFFFFFFF);
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
