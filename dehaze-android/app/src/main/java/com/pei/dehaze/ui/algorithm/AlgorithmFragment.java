package com.pei.dehaze.ui.algorithm;

import android.content.Intent;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;

import com.pei.dehaze.databinding.FragmentAlgorithmBinding;

public class AlgorithmFragment extends Fragment {

    private FragmentAlgorithmBinding binding;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        binding = FragmentAlgorithmBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        
        // 设置按钮点击事件
        binding.algorithmListButton.setOnClickListener(v -> {
            Intent intent = new Intent(getActivity(), AlgorithmListActivity.class);
            startActivity(intent);
        });
    }
    
    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}