package com.pei.dehaze.ui.presentation;

import android.os.Bundle;

import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.ViewModelProvider;

import com.pei.dehaze.R;
import com.pei.dehaze.ui.presentation.viewmodel.PresentationViewModel;

public class PresentationActivity extends AppCompatActivity {

    private PresentationViewModel presentationViewModel;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_presentation);

        initViewModel();
        setupObservers();
        loadData();
    }

    private void initViewModel() {
        presentationViewModel = new ViewModelProvider(this).get(PresentationViewModel.class);
    }

    private void setupObservers() {
        presentationViewModel.getLoading().observe(this, isLoading -> {
            // 处理加载状态
        });

        presentationViewModel.getError().observe(this, errorMessage -> {
            if (errorMessage != null && !errorMessage.isEmpty()) {
                // 显示错误信息
            }
        });
    }

    private void loadData() {
        // 加载算法列表
    }
}