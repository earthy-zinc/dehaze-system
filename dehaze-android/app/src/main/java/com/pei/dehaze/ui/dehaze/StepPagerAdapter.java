package com.pei.dehaze.ui.dehaze;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.SeekBar;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.RecyclerView;

import com.bumptech.glide.Glide;
import com.pei.dehaze.R;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.sdk.model.prediction.PredResult;

public class StepPagerAdapter extends RecyclerView.Adapter<RecyclerView.ViewHolder> {

    private final DehazeFragment fragment;

    public StepPagerAdapter(DehazeFragment fragment) {
        this.fragment = fragment;
    }

    @NonNull
    @Override
    public RecyclerView.ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        LayoutInflater inflater = LayoutInflater.from(parent.getContext());
        View view;
        switch (viewType) {
            case 0:
                view = inflater.inflate(R.layout.step_upload, parent, false);
                return new UploadViewHolder(view);
            case 1:
                view = inflater.inflate(R.layout.step_algorithm, parent, false);
                return new AlgorithmViewHolder(view);
            case 2:
                view = inflater.inflate(R.layout.step_params, parent, false);
                return new ParamsViewHolder(view);
            case 3:
                view = inflater.inflate(R.layout.step_process, parent, false);
                return new ProcessViewHolder(view);
            case 4:
                view = inflater.inflate(R.layout.step_compare, parent, false);
                return new CompareViewHolder(view);
            default:
                view = new View(parent.getContext());
                return new RecyclerView.ViewHolder(view) {};
        }
    }

    @Override
    public void onBindViewHolder(@NonNull RecyclerView.ViewHolder holder, int position) {
        DehazeViewModel vm = new ViewModelProvider(fragment).get(DehazeViewModel.class);

        if (holder instanceof UploadViewHolder) {
            UploadViewHolder vh = (UploadViewHolder) holder;
            vm.getUploadedFile().observe(fragment.getViewLifecycleOwner(), file -> {
                if (file != null && file.getUrl() != null) {
                    Glide.with(fragment).load(file.getUrl())
                            .placeholder(R.drawable.ic_image)
                            .error(R.drawable.ic_broken_image)
                            .into(vh.ivPreview);
                    vh.tvStatus.setText("已上传");
                }
            });
        } else if (holder instanceof AlgorithmViewHolder) {
            AlgorithmViewHolder vh = (AlgorithmViewHolder) holder;
            vm.getSelectedAlgorithmName().observe(fragment.getViewLifecycleOwner(), name -> {
                if (name != null) {
                    vh.tvSelectedAlgo.setText("已选择：" + name);
                    vh.btnSelectAlgo.setText("重新选择");
                }
            });
            // 绑定算法选择按钮
            vh.btnSelectAlgo.setOnClickListener(v -> fragment.launchAlgorithmSelect());
        } else if (holder instanceof ParamsViewHolder) {
            ParamsViewHolder vh = (ParamsViewHolder) holder;
            // 初始化滑块值
            Float strength = vm.getStrength().getValue();
            Float brightness = vm.getBrightness().getValue();
            Float contrast = vm.getContrast().getValue();
            if (strength != null) vh.sbStrength.setProgress((int)(strength * 100));
            if (brightness != null) vh.sbBrightness.setProgress((int)(brightness * 100));
            if (contrast != null) vh.sbContrast.setProgress((int)(contrast * 100));

            vh.sbStrength.setOnSeekBarChangeListener(new SimpleSeekBarListener() {
                @Override
                public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                    float value = progress / 100f;
                    vh.tvStrengthValue.setText(String.format("%.0f%%", value * 100));
                    vm.setStrength(value);
                }
            });
            vh.sbBrightness.setOnSeekBarChangeListener(new SimpleSeekBarListener() {
                @Override
                public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                    float value = progress / 100f;
                    vh.tvBrightnessValue.setText(String.format("%.0f%%", value * 100));
                    vm.setBrightness(value);
                }
            });
            vh.sbContrast.setOnSeekBarChangeListener(new SimpleSeekBarListener() {
                @Override
                public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                    float value = progress / 100f;
                    vh.tvContrastValue.setText(String.format("%.0f%%", value * 100));
                    vm.setContrast(value);
                }
            });
        } else if (holder instanceof ProcessViewHolder) {
            ProcessViewHolder vh = (ProcessViewHolder) holder;
            vm.getIsProcessing().observe(fragment.getViewLifecycleOwner(), isProcessing -> {
                boolean p = isProcessing != null && isProcessing;
                vh.pbProcess.setVisibility(p ? View.VISIBLE : View.GONE);
                vh.tvProcessStatus.setText(p ? "正在去雾处理中..." : "处理已完成");
            });
            vm.getPredictionResult().observe(fragment.getViewLifecycleOwner(), result -> {
                if (result != null) {
                    vh.tvProcessStatus.setText("处理完成！点击下一步查看对比");
                }
            });
        } else if (holder instanceof CompareViewHolder) {
            CompareViewHolder vh = (CompareViewHolder) holder;
            // 展示原始图和处理结果缩略图
            vm.getUploadedFile().observe(fragment.getViewLifecycleOwner(), file -> {
                if (file != null && file.getUrl() != null) {
                    Glide.with(fragment).load(file.getUrl())
                            .placeholder(R.drawable.ic_image)
                            .into(vh.ivOriginal);
                }
            });
            vm.getPredictionResult().observe(fragment.getViewLifecycleOwner(), result -> {
                if (result != null && result.getResultUrl() != null) {
                    Glide.with(fragment).load(result.getResultUrl())
                            .placeholder(R.drawable.ic_image)
                            .error(R.drawable.ic_broken_image)
                            .into(vh.ivResult);
                    vh.tvCompareHint.setText("处理完成，点击底部按钮查看详细对比");
                } else {
                    vh.tvCompareHint.setText("等待处理完成...");
                }
            });
        }
    }

    @Override
    public int getItemCount() {
        return 5;
    }

    @Override
    public int getItemViewType(int position) {
        return position;
    }

    static class UploadViewHolder extends RecyclerView.ViewHolder {
        final ImageView ivPreview;
        final TextView tvStatus;

        UploadViewHolder(View itemView) {
            super(itemView);
            ivPreview = itemView.findViewById(R.id.ivPreview);
            tvStatus = itemView.findViewById(R.id.tvStatus);
        }
    }

    static class AlgorithmViewHolder extends RecyclerView.ViewHolder {
        final TextView tvSelectedAlgo;
        final Button btnSelectAlgo;

        AlgorithmViewHolder(View itemView) {
            super(itemView);
            tvSelectedAlgo = itemView.findViewById(R.id.tvSelectedAlgo);
            btnSelectAlgo = itemView.findViewById(R.id.btnSelectAlgo);
        }
    }

    static class ParamsViewHolder extends RecyclerView.ViewHolder {
        final SeekBar sbStrength;
        final SeekBar sbBrightness;
        final SeekBar sbContrast;
        final TextView tvStrengthValue;
        final TextView tvBrightnessValue;
        final TextView tvContrastValue;

        ParamsViewHolder(View itemView) {
            super(itemView);
            sbStrength = itemView.findViewById(R.id.sbStrength);
            sbBrightness = itemView.findViewById(R.id.sbBrightness);
            sbContrast = itemView.findViewById(R.id.sbContrast);
            tvStrengthValue = itemView.findViewById(R.id.tvStrengthValue);
            tvBrightnessValue = itemView.findViewById(R.id.tvBrightnessValue);
            tvContrastValue = itemView.findViewById(R.id.tvContrastValue);
        }
    }

    static class ProcessViewHolder extends RecyclerView.ViewHolder {
        final ProgressBar pbProcess;
        final TextView tvProcessStatus;

        ProcessViewHolder(View itemView) {
            super(itemView);
            pbProcess = itemView.findViewById(R.id.pbProcess);
            tvProcessStatus = itemView.findViewById(R.id.tvProcessStatus);
        }
    }

    static class CompareViewHolder extends RecyclerView.ViewHolder {
        final ImageView ivOriginal;
        final ImageView ivResult;
        final TextView tvCompareHint;

        CompareViewHolder(View itemView) {
            super(itemView);
            ivOriginal = itemView.findViewById(R.id.ivOriginal);
            ivResult = itemView.findViewById(R.id.ivResult);
            tvCompareHint = itemView.findViewById(R.id.tvCompareHint);
        }
    }

    private static class SimpleSeekBarListener implements SeekBar.OnSeekBarChangeListener {
        @Override
        public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {}

        @Override
        public void onStartTrackingTouch(SeekBar seekBar) {}

        @Override
        public void onStopTrackingTouch(SeekBar seekBar) {}
    }
}
