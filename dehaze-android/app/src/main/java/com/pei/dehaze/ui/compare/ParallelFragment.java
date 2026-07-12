package com.pei.dehaze.ui.compare;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;

import com.bumptech.glide.Glide;
import com.pei.dehaze.R;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.sdk.model.prediction.PredResult;
import com.pei.dehaze.ui.compare.viewmodel.CompareViewModel;

import java.util.Map;

public class ParallelFragment extends Fragment {

    private CompareViewModel compareViewModel;
    private ImageView ivOriginal;
    private ImageView ivDehazed;
    private TextView tvAlgorithmInfo;

    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_parallel, container, false);
        ivOriginal = view.findViewById(R.id.iv_original);
        ivDehazed = view.findViewById(R.id.iv_dehazed);
        tvAlgorithmInfo = view.findViewById(R.id.tv_algorithm_info);
        return view;
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        compareViewModel = new ViewModelProvider(requireActivity()).get(CompareViewModel.class);

        compareViewModel.getUploadedFile().observe(getViewLifecycleOwner(), this::showOriginal);
        compareViewModel.getPredictionResult().observe(getViewLifecycleOwner(), this::showDehazed);
        compareViewModel.getMultiPredictionResults().observe(getViewLifecycleOwner(), this::showMultiResults);
    }

    private void showOriginal(FileInfo fileInfo) {
        if (fileInfo == null || fileInfo.getUrl() == null) return;
        Glide.with(this).load(fileInfo.getUrl())
                .placeholder(R.drawable.ic_image)
                .error(R.drawable.ic_broken_image)
                .into(ivOriginal);
    }

    private void showDehazed(PredResult result) {
        if (result == null) return;
        if (result.getResultUrl() == null) return;
        Glide.with(this).load(result.getResultUrl())
                .placeholder(R.drawable.ic_image)
                .error(R.drawable.ic_broken_image)
                .into(ivDehazed);
        tvAlgorithmInfo.setText("耗时：" + (result.getTime() == null ? "-" : result.getTime() + "ms")
                + (Boolean.TRUE.equals(result.getFromCache()) ? "（命中缓存）" : ""));
    }

    private void showMultiResults(Map<String, PredResult> results) {
        if (results == null || results.isEmpty()) return;
        PredResult first = results.values().iterator().next();
        if (first.getResultUrl() == null) return;
        Glide.with(this).load(first.getResultUrl())
                .placeholder(R.drawable.ic_image)
                .error(R.drawable.ic_broken_image)
                .into(ivDehazed);
        StringBuilder sb = new StringBuilder("完成 " + results.size() + " 个算法：");
        boolean firstItem = true;
        for (Map.Entry<String, PredResult> entry : results.entrySet()) {
            if (!firstItem) sb.append("、");
            sb.append("算法#").append(entry.getKey());
            firstItem = false;
        }
        tvAlgorithmInfo.setText(sb.toString());
    }
}
