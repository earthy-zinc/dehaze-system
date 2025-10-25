package com.pei.dehaze.ui.evaluation;

import android.os.Bundle;

import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.ViewModelProvider;

import com.pei.dehaze.R;
import com.pei.dehaze.ui.evaluation.viewmodel.EvaluationViewModel;

public class EvaluationActivity extends AppCompatActivity {

    private EvaluationViewModel evaluationViewModel;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_evaluation);

        initViewModel();
        setupObservers();
    }

    private void initViewModel() {
        evaluationViewModel = new ViewModelProvider(this).get(EvaluationViewModel.class);
    }

    private void setupObservers() {
        evaluationViewModel.getLoading().observe(this, isLoading -> {
            // 处理加载状态
        });

        evaluationViewModel.getError().observe(this, errorMessage -> {
            if (errorMessage != null && !errorMessage.isEmpty()) {
                // 显示错误信息
            }
        });
    }
}