package com.pei.clarity.ui.dehaze;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.SeekBar;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.bumptech.glide.Glide;
import com.pei.clarity.R;
import com.pei.clarity.sdk.model.file.FileInfo;
import com.pei.clarity.ui.common.adapter.ModelSelectAdapter;

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
                view = inflater.inflate(R.layout.step_model, parent, false);
                return new ModelViewHolder(view);
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
        } else if (holder instanceof ModelViewHolder) {
            ModelViewHolder vh = (ModelViewHolder) holder;
            ModelSelectAdapter modelAdapter = new ModelSelectAdapter(vm::setSelectedModel);
            vh.rvModels.setLayoutManager(new LinearLayoutManager(vh.itemView.getContext()));
            vh.rvModels.setAdapter(modelAdapter);

            vm.getModels().observe(fragment.getViewLifecycleOwner(), models -> {
                boolean empty = models == null || models.isEmpty();
                vh.tvModelEmpty.setVisibility(empty ? View.VISIBLE : View.GONE);
                modelAdapter.submitList(models);
            });
            vm.getLoading().observe(fragment.getViewLifecycleOwner(), loading -> {
                vh.tvModelEmpty.setText(Boolean.TRUE.equals(loading) ? "正在加载可用模型..." : "暂无可用模型");
            });
            vm.getSelectedModelId().observe(fragment.getViewLifecycleOwner(),
                    modelAdapter::setSelectedModelId);
            vm.getSelectedModelName().observe(fragment.getViewLifecycleOwner(), name -> {
                if (name != null) {
                    vh.tvSelectedModel.setText("已选择：" + name);
                }
            });
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

    static class ModelViewHolder extends RecyclerView.ViewHolder {
        final TextView tvSelectedModel;
        final TextView tvModelEmpty;
        final RecyclerView rvModels;

        ModelViewHolder(View itemView) {
            super(itemView);
            tvSelectedModel = itemView.findViewById(R.id.tvSelectedModel);
            tvModelEmpty = itemView.findViewById(R.id.tvModelEmpty);
            rvModels = itemView.findViewById(R.id.rvModels);
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
