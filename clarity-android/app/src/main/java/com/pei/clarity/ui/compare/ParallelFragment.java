package com.pei.clarity.ui.compare;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;

import com.bumptech.glide.Glide;
import com.pei.clarity.R;
import com.pei.clarity.databinding.FragmentParallelBinding;
import com.pei.clarity.sdk.model.file.FileInfo;
import com.pei.clarity.sdk.model.image_task.ImageTaskResponse;
import com.pei.clarity.ui.compare.viewmodel.CompareViewModel;
import com.pei.clarity.utils.StatePlaceholder;

import java.util.Map;

public class ParallelFragment extends Fragment {

    private CompareViewModel compareViewModel;
    private FragmentParallelBinding binding;
    private StatePlaceholder statePlaceholder;

    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        binding = FragmentParallelBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        compareViewModel = new ViewModelProvider(requireActivity()).get(CompareViewModel.class);

        statePlaceholder = new StatePlaceholder(binding.statePlaceholder.getRoot());
        statePlaceholder.showEmpty("请先完成去雾处理", R.drawable.ic_image_compare);

        compareViewModel.getUploadedFile().observe(getViewLifecycleOwner(), this::showOriginal);
        compareViewModel.getPredictionResult().observe(getViewLifecycleOwner(), this::showDehazed);
        compareViewModel.getMultiPredictionResults().observe(getViewLifecycleOwner(), this::showMultiResults);
        compareViewModel.getLoading().observe(getViewLifecycleOwner(), isLoading -> {
            if (isLoading != null && isLoading) {
                statePlaceholder.showLoading("正在处理中…");
            }
        });
    }

    private void showOriginal(FileInfo fileInfo) {
        if (fileInfo == null || fileInfo.getUrl() == null) return;
        Glide.with(this).load(fileInfo.getUrl())
                .placeholder(R.drawable.ic_image)
                .error(R.drawable.ic_broken_image)
                .into(binding.ivOriginal);
    }

    private void showDehazed(ImageTaskResponse result) {
        if (result == null || result.getResultUrl() == null) return;
        statePlaceholder.hide();
        Glide.with(this).load(result.getResultUrl())
                .placeholder(R.drawable.ic_image)
                .error(R.drawable.ic_broken_image)
                .into(binding.ivDehazed);
        binding.tvModelInfo.setText("耗时：" + (result.getTime() == null ? "-" : result.getTime() + "s"));
    }

    private void showMultiResults(Map<String, ImageTaskResponse> results) {
        if (results == null || results.isEmpty()) return;
        ImageTaskResponse first = results.values().iterator().next();
        if (first.getResultUrl() == null) return;
        statePlaceholder.hide();
        Glide.with(this).load(first.getResultUrl())
                .placeholder(R.drawable.ic_image)
                .error(R.drawable.ic_broken_image)
                .into(binding.ivDehazed);
        StringBuilder sb = new StringBuilder("完成 " + results.size() + " 个模型：");
        boolean firstItem = true;
        for (Map.Entry<String, ImageTaskResponse> entry : results.entrySet()) {
            if (!firstItem) sb.append("、");
            sb.append(entry.getKey());
            firstItem = false;
        }
        binding.tvModelInfo.setText(sb.toString());
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}
