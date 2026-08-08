package com.pei.dehaze.ui.dehaze;

import android.app.Activity;
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
import androidx.viewpager2.widget.ViewPager2;

import com.pei.dehaze.R;
import com.pei.dehaze.databinding.FragmentDehazeBinding;
import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.sdk.api.FileAPI;
import com.pei.dehaze.sdk.api.ModelAPI;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.sdk.model.prediction.DehazeParams;
import com.pei.dehaze.sdk.model.prediction.PredParam;
import com.pei.dehaze.sdk.model.prediction.PredResult;
import com.pei.dehaze.ui.algorithm_select.AlgorithmSelectActivity;
import com.pei.dehaze.ui.compare.CompareActivity;
import com.pei.dehaze.utils.ToastUtils;
import com.pei.dehaze.utils.UriUtils;

import java.io.File;

public class DehazeFragment extends Fragment {

    private static final int STEP_UPLOAD = 0;
    private static final int STEP_ALGORITHM = 1;
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

    private final ActivityResultLauncher<Intent> algorithmSelectLauncher =
            registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), result -> {
                if (result.getResultCode() == Activity.RESULT_OK && result.getData() != null) {
                    Intent data = result.getData();
                    long id = data.getLongExtra(AlgorithmSelectActivity.EXTRA_ALGORITHM_ID, 0L);
                    String name = data.getStringExtra(AlgorithmSelectActivity.EXTRA_ALGORITHM_NAME);
                    if (id > 0) {
                        dehazeViewModel.setSelectedAlgorithm(id, name != null ? name : "未知算法");
                        goToStep(STEP_PARAMS);
                    }
                }
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

        dehazeViewModel.getSelectedAlgorithmName().observe(getViewLifecycleOwner(), name -> {
            if (name != null) {
                ToastUtils.showShort(getContext(), "已选择算法：" + name);
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
            case STEP_ALGORITHM:
                launchAlgorithmSelect();
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
        binding.viewPager.setCurrentItem(step, true);
    }

    /**
     * 供 StepPagerAdapter 中算法选择按钮调用
     */
    public void launchAlgorithmSelect() {
        algorithmSelectLauncher.launch(new Intent(getActivity(), AlgorithmSelectActivity.class));
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
                        goToStep(STEP_ALGORITHM);
                    }

                    @Override
                    public void onError(String errorMessage) {
                        dehazeViewModel.setProcessing(false);
                        ToastUtils.showShort(getContext(), "上传失败: " + errorMessage);
                    }
                }));
    }

    private void startProcessing() {
        Long algorithmId = dehazeViewModel.getSelectedAlgorithmId().getValue();
        if (algorithmId == null) {
            ToastUtils.showShort(getContext(), "请先选择算法");
            goToStep(STEP_ALGORITHM);
            return;
        }

        new AlertDialog.Builder(requireContext())
                .setTitle("确认处理")
                .setMessage("确认开始去雾处理？")
                .setPositiveButton("确定", (d, w) -> {
                    dehazeViewModel.setProcessing(true);
                    // 使用实际参数值（DehazeParams 字段为 int 类型，范围 0-100 / 0-200）
                    DehazeParams params = new DehazeParams();
                    Float strength = dehazeViewModel.getStrength().getValue();
                    Float brightness = dehazeViewModel.getBrightness().getValue();
                    Float contrast = dehazeViewModel.getContrast().getValue();
                    params.setStrength(strength != null ? Math.round(strength) : 50);
                    params.setSaturation(brightness != null ? Math.round(brightness * 200) : 100);
                    params.setContrast(contrast != null ? Math.round(contrast * 200) : 100);

                    PredParam predParam = new PredParam();
                    predParam.setAlgorithmId(algorithmId);
                    predParam.setParams(params);
                    ModelAPI.predictAndWait(predParam, RepositoryAdapters.wrap(
                            new com.pei.dehaze.repository.RepositoryCallback<PredResult>() {
                                @Override
                                public void onSuccess(PredResult result) {
                                    dehazeViewModel.setProcessing(false);
                                    dehazeViewModel.setPredictionResult(result);
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
