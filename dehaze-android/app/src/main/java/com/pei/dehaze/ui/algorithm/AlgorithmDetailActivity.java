package com.pei.dehaze.ui.algorithm;

import android.app.AlertDialog;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.EditText;
import android.widget.Spinner;
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
import com.pei.dehaze.utils.ToastUtils;

import android.content.res.ColorStateList;

import java.util.ArrayList;
import java.util.List;

public class AlgorithmDetailActivity extends AppCompatActivity {

    private static final String[] STATUS_LABELS = {
            "草稿", "测试中", "待审核", "已发布", "已停用", "已归档"
    };
    private static final int[] STATUS_VALUES = {0, 1, 2, 3, 4, 5};

    private AlgorithmViewModel algorithmViewModel;
    private Toolbar toolbar;
    private TextView tvName, tvType, tvDescription, tvParams, tvFlops, tvSize, tvPath, tvImportPath;
    private Chip chipStatus;
    private MaterialButton btnEdit, btnToggleStatus, btnFavorite, btnDelete;

    private int algorithmId;
    private Algorithm currentAlgorithm;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_algorithm_detail);

        algorithmId = getIntent().getIntExtra("algorithm_id", 0);

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
        tvName.setText(safe(algorithm.getName()));
        tvType.setText(safe(algorithm.getType()));
        tvDescription.setText(safe(algorithm.getDescription()));
        tvParams.setText(safe(algorithm.getParams()));
        tvFlops.setText(safe(algorithm.getFlops()));
        tvSize.setText(safe(algorithm.getSize()));
        tvPath.setText(safe(algorithm.getPath()));
        tvImportPath.setText(safe(algorithm.getImportPath()));

        int statusValue = algorithm.getStatus() != null ? algorithm.getStatus() : 0;
        AlgorithmStatus status = AlgorithmStatus.fromValue(statusValue);
        chipStatus.setText(status.getLabel());
        chipStatus.setChipBackgroundColor(ColorStateList.valueOf(statusColor(statusValue)));
        chipStatus.setTextColor(0xFFFFFFFF);
    }

    private void showStatusTransitionDialog(Algorithm algorithm) {
        int currentStatus = algorithm.getStatus() != null ? algorithm.getStatus() : 0;
        List<Integer> nextStatuses = getNextStatuses(currentStatus);
        if (nextStatuses.isEmpty()) {
            ToastUtils.showShort(this, "当前状态「" + AlgorithmStatus.fromValue(currentStatus).getLabel() + "」不可流转");
            return;
        }
        String[] items = new String[nextStatuses.size()];
        for (int i = 0; i < nextStatuses.size(); i++) {
            items[i] = AlgorithmStatus.fromValue(nextStatuses.get(i)).getLabel();
        }
        new AlertDialog.Builder(this)
                .setTitle("状态流转 - " + safe(algorithm.getName()))
                .setItems(items, (dialog, which) -> {
                    int newStatus = nextStatuses.get(which);
                    algorithmViewModel.updateAlgorithmStatus(algorithm.getId(), newStatus);
                })
                .show();
    }

    private List<Integer> getNextStatuses(int currentStatus) {
        List<Integer> next = new ArrayList<>();
        switch (currentStatus) {
            case 0:
                next.add(1);
                break;
            case 1:
                next.add(2);
                break;
            case 2:
                next.add(3);
                next.add(1);
                break;
            case 3:
                next.add(4);
                break;
            case 4:
                next.add(3);
                next.add(5);
                break;
            case 5:
                break;
        }
        return next;
    }

    private void showDeleteConfirmDialog(Algorithm algorithm) {
        new AlertDialog.Builder(this)
                .setTitle("删除确认")
                .setMessage("确认删除算法「" + safe(algorithm.getName()) + "」吗？删除后不可恢复。")
                .setPositiveButton("确定", (dialog, which) -> {
                    algorithmViewModel.deleteAlgorithms(String.valueOf(algorithm.getId()));
                    finish();
                })
                .setNegativeButton("取消", null)
                .show();
    }

    private void showAlgorithmFormDialog(Algorithm existing) {
        View view = LayoutInflater.from(this).inflate(R.layout.dialog_algorithm_form, null);

        EditText etName = view.findViewById(R.id.et_name);
        EditText etType = view.findViewById(R.id.et_type);
        EditText etPath = view.findViewById(R.id.et_path);
        EditText etImportPath = view.findViewById(R.id.et_import_path);
        EditText etParams = view.findViewById(R.id.et_params);
        EditText etFlops = view.findViewById(R.id.et_flops);
        EditText etSize = view.findViewById(R.id.et_size);
        Spinner spinnerStatus = view.findViewById(R.id.spinner_status);
        EditText etDescription = view.findViewById(R.id.et_description);

        ArrayAdapter<String> statusAdapter = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_item, STATUS_LABELS);
        statusAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinnerStatus.setAdapter(statusAdapter);

        etName.setText(safe(existing.getName()));
        etType.setText(safe(existing.getType()));
        etPath.setText(safe(existing.getPath()));
        etImportPath.setText(safe(existing.getImportPath()));
        etParams.setText(safe(existing.getParams()));
        etFlops.setText(safe(existing.getFlops()));
        etSize.setText(safe(existing.getSize()));
        etDescription.setText(safe(existing.getDescription()));
        int currentStatus = existing.getStatus() != null ? existing.getStatus() : 0;
        for (int i = 0; i < STATUS_VALUES.length; i++) {
            if (STATUS_VALUES[i] == currentStatus) {
                spinnerStatus.setSelection(i);
                break;
            }
        }

        new AlertDialog.Builder(this)
                .setTitle("修改算法")
                .setView(view)
                .setPositiveButton("确定", (dialog, which) -> {
                    String name = etName.getText().toString().trim();
                    String type = etType.getText().toString().trim();
                    String path = etPath.getText().toString().trim();
                    String importPath = etImportPath.getText().toString().trim();
                    String params = etParams.getText().toString().trim();
                    String flops = etFlops.getText().toString().trim();
                    String size = etSize.getText().toString().trim();
                    String description = etDescription.getText().toString().trim();
                    int status = STATUS_VALUES[spinnerStatus.getSelectedItemPosition()];

                    if (TextUtils.isEmpty(name)) {
                        ToastUtils.showShort(this, "请输入算法名称");
                        return;
                    }
                    if (TextUtils.isEmpty(type)) {
                        ToastUtils.showShort(this, "请输入算法类型");
                        return;
                    }
                    if (TextUtils.isEmpty(path)) {
                        ToastUtils.showShort(this, "请输入模型文件路径");
                        return;
                    }
                    if (TextUtils.isEmpty(importPath)) {
                        ToastUtils.showShort(this, "请输入模型导入路径");
                        return;
                    }

                    Algorithm data = new Algorithm();
                    data.setId(existing.getId());
                    data.setParentId(existing.getParentId());
                    data.setName(name);
                    data.setType(type);
                    data.setPath(path);
                    data.setImportPath(importPath);
                    data.setParams(params);
                    data.setFlops(flops);
                    data.setSize(size);
                    data.setDescription(description);
                    data.setStatus(status);
                    algorithmViewModel.updateAlgorithm(existing.getId(), data);
                })
                .setNegativeButton("取消", null)
                .show();
    }

    private int statusColor(int status) {
        switch (status) {
            case 0: return 0xFF9E9E9E;
            case 1: return 0xFFFF9800;
            case 2: return 0xFF2196F3;
            case 3: return 0xFF4CAF50;
            case 4: return 0xFFE53935;
            case 5: return 0xFF607D8B;
            default: return 0xFF9E9E9E;
        }
    }

    private static String safe(String s) {
        return s == null ? "" : s;
    }
}
