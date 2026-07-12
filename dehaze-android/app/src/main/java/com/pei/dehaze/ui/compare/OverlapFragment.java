package com.pei.dehaze.ui.compare;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.SeekBar;
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

public class OverlapFragment extends Fragment {

    private CompareViewModel compareViewModel;
    private ImageView ivOriginal;
    private ImageView ivDehazed;
    private SeekBar seekBarOverlap;
    private TextView tvOverlapValue;

    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_overlap, container, false);
        ivOriginal = view.findViewById(R.id.iv_original);
        ivDehazed = view.findViewById(R.id.iv_dehazed);
        seekBarOverlap = view.findViewById(R.id.seek_bar_overlap);
        tvOverlapValue = view.findViewById(R.id.tv_overlap_value);
        return view;
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        compareViewModel = new ViewModelProvider(requireActivity()).get(CompareViewModel.class);

        compareViewModel.getUploadedFile().observe(getViewLifecycleOwner(), this::showOriginal);
        compareViewModel.getPredictionResult().observe(getViewLifecycleOwner(), this::showDehazed);

        seekBarOverlap.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                float alpha = progress / 100f;
                ivDehazed.setAlpha(alpha);
                tvOverlapValue.setText("去雾图透明度：" + progress + "%");
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
            }
        });
        ivDehazed.setAlpha(seekBarOverlap.getProgress() / 100f);
    }

    private void showOriginal(FileInfo fileInfo) {
        if (fileInfo == null || fileInfo.getUrl() == null) return;
        Glide.with(this).load(fileInfo.getUrl())
                .placeholder(R.drawable.ic_image)
                .error(R.drawable.ic_broken_image)
                .into(ivOriginal);
    }

    private void showDehazed(PredResult result) {
        if (result == null || result.getResultUrl() == null) return;
        Glide.with(this).load(result.getResultUrl())
                .placeholder(R.drawable.ic_image)
                .error(R.drawable.ic_broken_image)
                .into(ivDehazed);
    }
}
