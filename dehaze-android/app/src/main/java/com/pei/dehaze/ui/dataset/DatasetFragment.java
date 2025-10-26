package com.pei.dehaze.ui.dataset;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;

import com.pei.dehaze.R;
import com.pei.dehaze.databinding.FragmentDatasetBinding;

public class DatasetFragment extends Fragment {

    private DatasetViewModel datasetViewModel;
    private FragmentDatasetBinding binding;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        binding = FragmentDatasetBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        
        datasetViewModel = new ViewModelProvider(this).get(DatasetViewModel.class);
        binding.setLifecycleOwner(this);
        binding.setViewModel(datasetViewModel);
        
        setupObservers();
        loadData();
    }

    private void setupObservers() {
        datasetViewModel.getLoading().observe(getViewLifecycleOwner(), isLoading -> {
            // 处理加载状态
        });

        datasetViewModel.getError().observe(getViewLifecycleOwner(), errorMessage -> {
            if (errorMessage != null && !errorMessage.isEmpty()) {
                // 显示错误信息
            }
        });
    }

    private void loadData() {
        // 先加载数据集信息
        datasetViewModel.loadDatasetInfo();
        // 再加载图片数据
        datasetViewModel.loadImages();
    }
    
    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}