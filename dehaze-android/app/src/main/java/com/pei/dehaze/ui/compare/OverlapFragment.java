package com.pei.dehaze.ui.compare;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.SeekBar;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;

import com.bumptech.glide.Glide;
import com.pei.dehaze.R;
import com.pei.dehaze.databinding.FragmentOverlapBinding;
import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.sdk.model.prediction.PredResult;
import com.pei.dehaze.ui.compare.viewmodel.CompareViewModel;
import com.pei.dehaze.utils.StatePlaceholder;

public class OverlapFragment extends Fragment {

    private CompareViewModel compareViewModel;
    private FragmentOverlapBinding binding;
    private StatePlaceholder statePlaceholder;

    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        binding = FragmentOverlapBinding.inflate(inflater, container, false);
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
        compareViewModel.getLoading().observe(getViewLifecycleOwner(), isLoading -> {
            if (isLoading != null && isLoading) {
                statePlaceholder.showLoading("正在处理中…");
            }
        });

        binding.seekBarOverlap.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                float alpha = progress / 100f;
                binding.ivDehazed.setAlpha(alpha);
                binding.tvOverlapValue.setText("去雾图透明度：" + progress + "%");
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
            }
        });
        binding.ivDehazed.setAlpha(binding.seekBarOverlap.getProgress() / 100f);
    }

    private void showOriginal(FileInfo fileInfo) {
        if (fileInfo == null || fileInfo.getUrl() == null) return;
        Glide.with(this).load(DehazeSDK.getInstance().resolveUrl(fileInfo.getUrl()))
                .placeholder(R.drawable.ic_image)
                .error(R.drawable.ic_broken_image)
                .into(binding.ivOriginal);
    }

    private void showDehazed(PredResult result) {
        if (result == null || result.getResultUrl() == null) return;
        statePlaceholder.hide();
        Glide.with(this).load(DehazeSDK.getInstance().resolveUrl(result.getResultUrl()))
                .placeholder(R.drawable.ic_image)
                .error(R.drawable.ic_broken_image)
                .into(binding.ivDehazed);
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}
