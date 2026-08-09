package com.pei.dehaze.ui.algorithm;

import android.content.Intent;
import android.content.res.ColorStateList;
import android.os.Bundle;

import androidx.lifecycle.ViewModelProvider;

import com.pei.dehaze.databinding.ActivityAlgorithmDetailBinding;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmStatus;
import com.pei.dehaze.ui.algorithm.viewmodel.AlgorithmBrowseViewModel;
import com.pei.dehaze.ui.algorithm.viewmodel.AlgorithmViewModel;
import com.pei.dehaze.ui.algorithm_select.AlgorithmSelectActivity;
import com.pei.dehaze.ui.algorithm_select.viewmodel.AlgorithmSelectViewModel;
import com.pei.dehaze.utils.StringUtils;
import com.pei.dehaze.utils.ToastUtils;
import com.pei.dehaze.ui.common.BaseActivity;

/**
 * 算法详情（浏览版）— 查看详情 + 收藏 + 使用该算法
 */
public class AlgorithmDetailActivity extends BaseActivity {

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
        setupToolbar(binding.toolbar, null);

        binding.btnUse.setOnClickListener(v -> {
            if (currentAlgorithm != null) {
                Intent intent = new Intent(this, AlgorithmSelectActivity.class);
                intent.putExtra(AlgorithmSelectActivity.EXTRA_ALGORITHM_ID, currentAlgorithm.getId());
                intent.putExtra(AlgorithmSelectActivity.EXTRA_ALGORITHM_NAME, currentAlgorithm.getName());
                startActivity(intent);
            }
        });

        binding.btnFavorite.setOnClickListener(v -> {
            if (currentAlgorithm != null) {
                selectViewModel.addFavorite(currentAlgorithm.getId());
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

        observeError(algorithmViewModel);

        observeOperationResult(selectViewModel, null);

        observeError(selectViewModel);
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
