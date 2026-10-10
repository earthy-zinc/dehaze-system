package com.pei.dehaze.ui.dehaze;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;

import com.pei.dehaze.R;
import com.pei.dehaze.databinding.FragmentDehazeBinding;
import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.sdk.api.AiImageAPI;
import com.pei.dehaze.sdk.api.FileAPI;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.sdk.model.image_task.DehazeParams;
import com.pei.dehaze.sdk.model.image_task.ImageEditForm;
import com.pei.dehaze.sdk.model.image_task.ImageTaskResponse;
import com.pei.dehaze.sdk.model.image_task.ImageTaskStatus;
import com.pei.dehaze.ui.compare.CompareActivity;
import com.pei.dehaze.utils.ToastUtils;
import com.pei.dehaze.utils.UriUtils;

import java.io.File;

public class DehazeFragment extends Fragment {

    private static final int STEP_UPLOAD = 0;
    private static final int STEP_MODEL = 1;
    private static final int STEP_PARAMS = 2;
    private static final int STEP_PROCESS = 3;
    private static final int STEP_COMPARE = 4;

    private DehazeViewModel dehazeViewModel;
    private FragmentDehazeBinding binding;
    private TextView[] stepViews;

    private final ActivityResultLauncher<String> pickImageLauncher =
            registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
                if (uri != null) uploadImage(uri);
            });

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        binding = FragmentDehazeBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        dehazeViewModel = new ViewModelProvider(this).get(DehazeViewModel.class);

        stepViews = new TextView[]{
                binding.tvStep1, binding.tvStep2, binding.tvStep3, binding.tvStep4, binding.tvStep5
        };

        initViews();
        setupObservers();
        dehazeViewModel.loadModels();
    }

    private void initViews() {
        binding.viewPager.setUserInputEnabled(false);
        binding.viewPager.setOffscreenPageLimit(5);
        binding.viewPager.setAdapter(new StepPagerAdapter(this));

        binding.btnNext.setOnClickListener(v -> onNextStep());
        binding.btnPrev.setOnClickListener(v -> onPrevStep());

        updateStepIndicator(0);
    }

    private void setupObservers() {
        dehazeViewModel.getCurrentStep().observe(getViewLifecycleOwner(), step -> {
            if (step != null) {
                updateStepIndicator(step);
                updateButtons(step);
            }
        });

        dehazeViewModel.getSelectedModelName().observe(getViewLifecycleOwner(), name -> {
            if (name != null) {
                ToastUtils.showShort(getContext(), "已选择模型：" + name);
            }
        });

        dehazeViewModel.getIsProcessing().observe(getViewLifecycleOwner(), isProcessing -> {
            boolean p = isProcessing != null && isProcessing;
            binding.btnNext.setEnabled(!p);
            binding.btnNext.setText(p ? "处理中..." : "下一步");
        });
    }

    private void updateStepIndicator(int current) {
        for (int i = 0; i < stepViews.length; i++) {
            if (i < current) {
                stepViews[i].setBackgroundResource(R.drawable.bg_metric_gradient);
                stepViews[i].setTextColor(getResources().getColor(R.color.white, null));
            } else if (i == current) {
                stepViews[i].setBackgroundResource(R.drawable.bg_primary_gradient);
                stepViews[i].setTextColor(getResources().getColor(R.color.white, null));
            } else {
                stepViews[i].setBackgroundColor(getResources().getColor(R.color.bg_page, null));
                stepViews[i].setTextColor(getResources().getColor(R.color.text_secondary, null));
            }
        }
    }

    private void updateButtons(int step) {
        if (step == 0) {
            binding.btnPrev.setVisibility(View.GONE);
            binding.btnNext.setText("上传图片");
        } else if (step == 4) {
            binding.btnPrev.setVisibility(View.VISIBLE);
            binding.btnNext.setText("查看对比");
        } else {
            binding.btnPrev.setVisibility(View.VISIBLE);
            binding.btnNext.setText("下一步");
        }
    }

    private void onNextStep() {
        int step = dehazeViewModel.getCurrentStep().getValue() != null
                ? dehazeViewModel.getCurrentStep().getValue() : 0;

        switch (step) {
            case STEP_UPLOAD:
                pickImageLauncher.launch("image/*");
                break;
            case STEP_MODEL:
                if (dehazeViewModel.getSelectedModelId().getValue() == null) {
                    ToastUtils.showShort(getContext(), "请先选择去雾模型");
                    return;
                }
                goToStep(STEP_PARAMS);
                break;
            case STEP_PARAMS:
                goToStep(STEP_PROCESS);
                startProcessing();
                break;
            case STEP_PROCESS:
                break;
            case STEP_COMPARE:
                startActivity(new Intent(getActivity(), CompareActivity.class));
                break;
            default:
                break;
        }
    }

    private void onPrevStep() {
        int step = dehazeViewModel.getCurrentStep().getValue() != null
                ? dehazeViewModel.getCurrentStep().getValue() : 0;
        if (step > 0) {
            goToStep(step - 1);
        }
    }

    private void goToStep(int step) {
        dehazeViewModel.setCurrentStep(step);
    }

    private void uploadImage(Uri uri) {
        File tempFile = UriUtils.copyToCache(requireContext(), uri);
        if (tempFile == null) {
            ToastUtils.showShort(getContext(), "无法读取所选图片");
            return;
        }
        dehazeViewModel.setProcessing(true);
        FileAPI.upload(tempFile, RepositoryAdapters.wrap(
                new com.pei.dehaze.repository.RepositoryCallback<FileInfo>() {
                    @Override
                    public void onSuccess(FileInfo data) {
                        dehazeViewModel.setUploadedFile(data);
                        dehazeViewModel.setProcessing(false);
                        goToStep(STEP_MODEL);
                    }

                    @Override
                    public void onError(String errorMessage) {
                        dehazeViewModel.setProcessing(false);
                        ToastUtils.showShort(getContext(), "上传失败: " + errorMessage);
                    }
                }));
    }

    private void startProcessing() {
        String modelId = dehazeViewModel.getSelectedModelId().getValue();
        FileInfo uploaded = dehazeViewModel.getUploadedFile().getValue();
        if (modelId == null) {
            ToastUtils.showShort(getContext(), "请先选择去雾模型");
            goToStep(STEP_MODEL);
            return;
        }
        if (uploaded == null || uploaded.getId() == null) {
            ToastUtils.showShort(getContext(), "请先上传图片");
            goToStep(STEP_UPLOAD);
            return;
        }

        new AlertDialog.Builder(requireContext())
                .setTitle("确认处理")
                .setMessage("确认开始去雾处理？")
                .setPositiveButton("确定", (d, w) -> {
                    dehazeViewModel.setProcessing(true);
                    // 滑块为 0-1 归一化值，映射到参数实际范围：强度 0-100、饱和度/对比度 0-200
                    DehazeParams params = new DehazeParams();
                    Float strength = dehazeViewModel.getStrength().getValue();
                    Float brightness = dehazeViewModel.getBrightness().getValue();
                    Float contrast = dehazeViewModel.getContrast().getValue();
                    params.setStrength(strength != null ? Math.round(strength * 100) : 50);
                    params.setSaturation(brightness != null ? Math.round(brightness * 200) : 100);
                    params.setContrast(contrast != null ? Math.round(contrast * 200) : 100);

                    ImageEditForm form = new ImageEditForm();
                    form.setModel(modelId);
                    form.setFileId(uploaded.getId());
                    form.setParams(params);

                    AiImageAPI.submitAndWait(form, RepositoryAdapters.wrap(
                            new com.pei.dehaze.repository.RepositoryCallback<ImageTaskResponse>() {
                                @Override
                                public void onSuccess(ImageTaskResponse result) {
                                    dehazeViewModel.setProcessing(false);
                                    dehazeViewModel.setPredictionResult(result);
                                    if (result == null || result.getStatus() == null) {
                                        ToastUtils.showShort(getContext(), "处理结果异常，请稍后重试");
                                        return;
                                    }
                                    if (result.getStatus() == ImageTaskStatus.FAILED) {
                                        ToastUtils.showShort(getContext(), "处理失败: "
                                                + (result.getErrorMessage() == null ? "未知错误" : result.getErrorMessage()));
                                        return;
                                    }
                                    if (result.getStatus() == ImageTaskStatus.CANCELED) {
                                        ToastUtils.showShort(getContext(), "任务已取消");
                                        return;
                                    }
                                    ToastUtils.showShort(getContext(), "处理完成！");
                                    goToStep(STEP_COMPARE);
                                }

                                @Override
                                public void onError(String errorMessage) {
                                    dehazeViewModel.setProcessing(false);
                                    ToastUtils.showShort(getContext(), "处理失败: " + errorMessage);
                                }
                            }));
                })
                .setNegativeButton("取消", null)
                .show();
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}
