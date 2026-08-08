package com.pei.dehaze.ui.compare;

import android.graphics.Bitmap;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
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
import com.bumptech.glide.request.target.CustomTarget;
import com.bumptech.glide.request.transition.Transition;
import com.pei.dehaze.R;
import com.pei.dehaze.databinding.FragmentFilterBinding;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.sdk.model.prediction.PredResult;
import com.pei.dehaze.ui.compare.viewmodel.CompareViewModel;
import com.pei.dehaze.utils.StatePlaceholder;

/**
 * 滤镜对比 Fragment：亮度/对比度/饱和度三档滑杆实时调节去雾图，与原图并排对比。
 * 通过 ColorMatrix 组合实现调色。
 */
public class FilterFragment extends Fragment {

    private CompareViewModel compareViewModel;
    private FragmentFilterBinding binding;
    private StatePlaceholder statePlaceholder;

    private float brightness = 0f;    // -100 ~ 100
    private float contrast = 1f;     // 0 ~ 2（1 = 原图）
    private float saturation = 1f;   // 0 ~ 2（1 = 原图）

    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        binding = FragmentFilterBinding.inflate(inflater, container, false);
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

        binding.seekBrightness.setOnSeekBarChangeListener(new SimpleSeekListener(progress -> {
            brightness = progress - 100f;
            binding.tvBrightnessValue.setText(String.format("亮度：%d", (int) brightness));
            applyFilter();
        }));
        binding.seekContrast.setOnSeekBarChangeListener(new SimpleSeekListener(progress -> {
            contrast = progress / 50f;
            binding.tvContrastValue.setText(String.format("对比度：%.2f", contrast));
            applyFilter();
        }));
        binding.seekSaturation.setOnSeekBarChangeListener(new SimpleSeekListener(progress -> {
            saturation = progress / 50f;
            binding.tvSaturationValue.setText(String.format("饱和度：%.2f", saturation));
            applyFilter();
        }));
        binding.btnReset.setOnClickListener(v -> {
            binding.seekBrightness.setProgress(50);
            binding.seekContrast.setProgress(50);
            binding.seekSaturation.setProgress(50);
        });
    }

    private void showOriginal(FileInfo fileInfo) {
        if (fileInfo == null || fileInfo.getUrl() == null) return;
        Glide.with(this).asBitmap().load(fileInfo.getUrl())
                .placeholder(R.drawable.ic_image)
                .error(R.drawable.ic_broken_image)
                .into(new CustomTarget<Bitmap>() {
                    @Override
                    public void onResourceReady(@NonNull Bitmap resource, @Nullable Transition<? super Bitmap> transition) {
                        binding.ivOriginal.setImageBitmap(resource);
                    }

                    @Override
                    public void onLoadCleared(@Nullable android.graphics.drawable.Drawable placeholder) {}
                });
    }

    private void showDehazed(PredResult result) {
        if (result == null || result.getResultUrl() == null) return;
        statePlaceholder.hide();
        Glide.with(this).asBitmap().load(result.getResultUrl())
                .placeholder(R.drawable.ic_image)
                .error(R.drawable.ic_broken_image)
                .into(new CustomTarget<Bitmap>() {
                    @Override
                    public void onResourceReady(@NonNull Bitmap resource, @Nullable Transition<? super Bitmap> transition) {
                        binding.ivDehazed.setImageBitmap(resource);
                        applyFilter();
                    }

                    @Override
                    public void onLoadCleared(@Nullable android.graphics.drawable.Drawable placeholder) {}
                });
    }

    private void applyFilter() {
        if (binding.ivDehazed.getDrawable() == null) return;

        // 1. 饱和度矩阵
        ColorMatrix satMatrix = new ColorMatrix();
        satMatrix.setSaturation(saturation);

        // 2. 对比度矩阵
        ColorMatrix contrastMatrix = new ColorMatrix(new float[]{
                contrast, 0, 0, 0, 0,
                0, contrast, 0, 0, 0,
                0, 0, contrast, 0, 0,
                0, 0, 0, 1, 0
        });

        // 3. 亮度矩阵（RGB 加偏移）
        ColorMatrix brightMatrix = new ColorMatrix(new float[]{
                1, 0, 0, 0, brightness,
                0, 1, 0, 0, brightness,
                0, 0, 1, 0, brightness,
                0, 0, 0, 1, 0
        });

        // 组合：sat × contrast × bright
        contrastMatrix.postConcat(satMatrix);
        brightMatrix.postConcat(contrastMatrix);

        binding.ivDehazed.setColorFilter(new ColorMatrixColorFilter(brightMatrix));
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }

    private static class SimpleSeekListener implements SeekBar.OnSeekBarChangeListener {
        interface OnProgress {
            void onProgress(int progress);
        }

        private final OnProgress callback;

        SimpleSeekListener(OnProgress callback) {
            this.callback = callback;
        }

        @Override
        public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
            callback.onProgress(progress);
        }

        @Override
        public void onStartTrackingTouch(SeekBar seekBar) {}

        @Override
        public void onStopTrackingTouch(SeekBar seekBar) {}
    }
}
