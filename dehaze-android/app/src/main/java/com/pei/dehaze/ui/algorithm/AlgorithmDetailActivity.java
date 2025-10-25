package com.pei.dehaze.ui.algorithm;

import android.os.Bundle;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.ViewModelProvider;

import com.pei.dehaze.R;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.ui.algorithm.viewmodel.AlgorithmViewModel;

public class AlgorithmDetailActivity extends AppCompatActivity {

    private AlgorithmViewModel algorithmViewModel;
    private TextView tvName, tvType, tvDescription, tvParams, tvFlops, tvSize;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_algorithm_detail);

        initViews();
        initViewModel();
        setupObservers();

        int algorithmId = getIntent().getIntExtra("algorithm_id", 0);
        if (algorithmId > 0) {
            algorithmViewModel.loadAlgorithmDetail(algorithmId);
        }
    }

    private void initViews() {
        tvName = findViewById(R.id.tv_algorithm_name);
        tvType = findViewById(R.id.tv_algorithm_type);
        tvDescription = findViewById(R.id.tv_algorithm_description);
        tvParams = findViewById(R.id.tv_algorithm_params);
        tvFlops = findViewById(R.id.tv_algorithm_flops);
        tvSize = findViewById(R.id.tv_algorithm_size);
    }

    private void initViewModel() {
        algorithmViewModel = new ViewModelProvider(this).get(AlgorithmViewModel.class);
    }

    private void setupObservers() {
        algorithmViewModel.getAlgorithmDetail().observe(this, algorithm -> {
            if (algorithm != null) {
                updateUI(algorithm);
            }
        });

        algorithmViewModel.getError().observe(this, errorMessage -> {
            if (errorMessage != null && !errorMessage.isEmpty()) {
                // 显示错误信息
            }
        });
    }

    private void updateUI(Algorithm algorithm) {
        tvName.setText(algorithm.getName());
        tvType.setText(algorithm.getType());
        tvDescription.setText(algorithm.getDescription());
        tvParams.setText(algorithm.getParams());
        tvFlops.setText(algorithm.getFlops());
        tvSize.setText(algorithm.getSize());
    }
}