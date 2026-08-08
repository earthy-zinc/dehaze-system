package com.pei.dehaze.ui.compare;

import android.net.Uri;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;

import com.pei.dehaze.R;
import com.pei.dehaze.databinding.FragmentMetricsBinding;
import com.pei.dehaze.sdk.model.evaluation.EvalResult;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.sdk.model.prediction.PredResult;
import com.pei.dehaze.ui.compare.viewmodel.CompareViewModel;
import com.pei.dehaze.utils.StatePlaceholder;
import com.pei.dehaze.utils.ToastUtils;
import com.pei.dehaze.utils.UriUtils;

import java.io.File;
import java.util.Map;

/**
 * 指标评估 Fragment：用户上传参考图（GT）后，调 ModelAPI.evaluateAndWait
 * 计算去雾结果与 GT 的 PSNR/SSIM 等客观指标，以表格形式展示。
 */
public class MetricsFragment extends Fragment {

    private CompareViewModel compareViewModel;
    private FragmentMetricsBinding binding;
    private StatePlaceholder statePlaceholder;

    private PredResult currentPredResult;
    private String gtUrl;

    private final ActivityResultLauncher<String> pickGtLauncher =
            registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
                if (uri != null) uploadGt(uri);
            });

    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        binding = FragmentMetricsBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        compareViewModel = new ViewModelProvider(requireActivity()).get(CompareViewModel.class);

        statePlaceholder = new StatePlaceholder(binding.statePlaceholder.getRoot());
        statePlaceholder.showEmpty("请先完成去雾处理", R.drawable.ic_metrics);

        compareViewModel.getPredictionResult().observe(getViewLifecycleOwner(), result -> {
            currentPredResult = result;
            if (result != null && result.getResultUrl() != null) {
                statePlaceholder.showEmpty("上传参考图（GT）以计算指标", R.drawable.ic_metrics);
                binding.btnUploadGt.setEnabled(true);
            }
        });
        compareViewModel.getEvaluationResult().observe(getViewLifecycleOwner(), this::showMetrics);
        compareViewModel.getLoading().observe(getViewLifecycleOwner(), isLoading -> {
            if (isLoading != null && isLoading) {
                statePlaceholder.showLoading("正在评估…");
            } else if (currentPredResult != null) {
                statePlaceholder.hide();
            }
        });

        binding.btnUploadGt.setEnabled(false);
        binding.btnUploadGt.setOnClickListener(v -> {
            if (currentPredResult == null || currentPredResult.getResultUrl() == null) {
                ToastUtils.showShort(requireContext(), "请先完成去雾处理");
                return;
            }
            pickGtLauncher.launch("image/*");
        });
        binding.btnEvaluate.setOnClickListener(v -> onEvaluateClick());
    }

    private void uploadGt(Uri uri) {
        File tempFile = UriUtils.copyToCache(requireContext(), uri);
        if (tempFile == null) {
            ToastUtils.showShort(requireContext(), "无法读取所选参考图");
            return;
        }
        compareViewModel.uploadGtImage(tempFile);
        // 监听 GT 上传成功事件
        compareViewModel.getUploadedGtFile().observe(getViewLifecycleOwner(), fileInfo -> {
            if (fileInfo != null && fileInfo.getUrl() != null) {
                gtUrl = fileInfo.getUrl();
                binding.tvGtStatus.setText("参考图已上传");
                binding.btnEvaluate.setEnabled(true);
            }
        });
    }

    private void onEvaluateClick() {
        if (currentPredResult == null || currentPredResult.getResultUrl() == null) {
            ToastUtils.showShort(requireContext(), "请先完成去雾处理");
            return;
        }
        if (TextUtils.isEmpty(gtUrl)) {
            ToastUtils.showShort(requireContext(), "请先上传参考图");
            return;
        }
        Long algorithmId = compareViewModel.getCurrentAlgorithmId();
        if (algorithmId == null) {
            ToastUtils.showShort(requireContext(), "算法 ID 缺失");
            return;
        }
        compareViewModel.evaluate(algorithmId, currentPredResult.getResultUrl(), gtUrl);
    }

    private void showMetrics(EvalResult result) {
        if (result == null || result.getMetrics() == null || result.getMetrics().isEmpty()) {
            statePlaceholder.showEmpty("未获取到评估指标", R.drawable.ic_metrics);
            return;
        }
        statePlaceholder.hide();
        binding.metricsContainer.removeAllViews();

        Map<String, Double> metrics = result.getMetrics();
        for (Map.Entry<String, Double> entry : metrics.entrySet()) {
            View row = LayoutInflater.from(requireContext())
                    .inflate(R.layout.item_metric_row, binding.metricsContainer, false);
            TextView tvName = row.findViewById(R.id.tv_metric_name);
            TextView tvValue = row.findViewById(R.id.tv_metric_value);
            tvName.setText(entry.getKey());
            tvValue.setText(String.format("%.4f", entry.getValue()));
            binding.metricsContainer.addView(row);
        }

        if (result.getTime() != null) {
            binding.tvEvalTime.setText("耗时：" + result.getTime() + "ms");
            binding.tvEvalTime.setVisibility(View.VISIBLE);
        }
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}
