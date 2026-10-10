package com.pei.dehaze.ui.evaluation;

import androidx.appcompat.app.AlertDialog;
import android.net.Uri;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.View;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import com.pei.dehaze.ui.common.BaseActivity;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;

import com.bumptech.glide.Glide;
import com.pei.dehaze.R;
import com.pei.dehaze.databinding.ActivityEvaluationBinding;
import com.pei.dehaze.sdk.model.ai_model.AiModelVO;
import com.pei.dehaze.sdk.model.evaluation.EvalResult;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.sdk.model.image_task.DehazeParams;
import com.pei.dehaze.sdk.model.evaluation.EvalTaskStatus;
import com.pei.dehaze.sdk.model.image_task.ImageTaskResponse;
import com.pei.dehaze.ui.evaluation.adapter.EvaluationLogAdapter;
import com.pei.dehaze.ui.evaluation.adapter.MetricAdapter;
import com.pei.dehaze.ui.evaluation.viewmodel.EvaluationViewModel;
import com.pei.dehaze.utils.StatePlaceholder;
import com.pei.dehaze.utils.StringUtils;
import com.pei.dehaze.utils.ToastUtils;
import com.pei.dehaze.utils.UriUtils;
import com.pei.dehaze.utils.ViewUtils;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class EvaluationActivity extends BaseActivity {

    private EvaluationViewModel evaluationViewModel;
    private ActivityEvaluationBinding binding;
    private StatePlaceholder statePlaceholder;

    private final List<AiModelVO> modelOptions = new ArrayList<>();
    private MetricAdapter metricAdapter;
    private EvaluationLogAdapter evaluationLogAdapter;

    private final ActivityResultLauncher<String> pickHazyLauncher =
            registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
                if (uri != null) uploadHazyImage(uri);
            });

    private final ActivityResultLauncher<String> pickClearLauncher =
            registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
                if (uri != null) uploadClearImage(uri);
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityEvaluationBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        initViews();
        initViewModel();
        setupObservers();
        loadData();
    }

    private void initViews() {
        setSupportActionBar(binding.toolbar);
        binding.toolbar.setNavigationOnClickListener(v -> finish());

        binding.btnSelectHazy.setOnClickListener(v -> pickHazyLauncher.launch("image/*"));
        binding.btnSelectClear.setOnClickListener(v -> pickClearLauncher.launch("image/*"));

        binding.btnPredict.setOnClickListener(v -> onPredictClick());
        binding.btnEvaluate.setOnClickListener(v -> onEvaluateClick());

        metricAdapter = new MetricAdapter();
        binding.rvMetrics.setLayoutManager(new LinearLayoutManager(this));
        binding.rvMetrics.setAdapter(metricAdapter);

        evaluationLogAdapter = new EvaluationLogAdapter();
        binding.rvHistory.setLayoutManager(new LinearLayoutManager(this));
        binding.rvHistory.setAdapter(evaluationLogAdapter);

        statePlaceholder = new StatePlaceholder(binding.statePlaceholder.getRoot());
        statePlaceholder.showEmpty("请先执行去雾处理", R.drawable.ic_image);
        binding.tvMetricsEmpty.setVisibility(View.VISIBLE);
    }

    private void onPredictClick() {
        String modelId = getCurrentModelId();
        if (modelId == null) {
            ToastUtils.showShort(this, "请先选择模型");
            return;
        }
        new AlertDialog.Builder(this)
                .setTitle("确认处理")
                .setMessage("确认开始去雾处理？")
                .setPositiveButton("确定", (d, w) -> evaluationViewModel.predict(modelId, new DehazeParams()))
                .setNegativeButton("取消", null)
                .show();
    }

    private void onEvaluateClick() {
        String modelId = getCurrentModelId();
        if (modelId == null) {
            ToastUtils.showShort(this, "请先选择模型");
            return;
        }
        new AlertDialog.Builder(this)
                .setTitle("确认评估")
                .setMessage("确认开始指标评估？")
                .setPositiveButton("确定", (d, w) -> evaluationViewModel.evaluate(modelId))
                .setNegativeButton("取消", null)
                .show();
    }

    private String getCurrentModelId() {
        int pos = binding.spinnerModel.getSelectedItemPosition();
        if (pos < 0 || pos >= modelOptions.size()) return null;
        return modelOptions.get(pos).getModelId();
    }

    private void initViewModel() {
        evaluationViewModel = new ViewModelProvider(this).get(EvaluationViewModel.class);
    }

    private void setupObservers() {
        evaluationViewModel.getHazyFile().observe(this, this::showHazyImage);
        evaluationViewModel.getClearFile().observe(this, this::showClearImage);
        evaluationViewModel.getModels().observe(this, this::updateModelSpinner);
        evaluationViewModel.getPredictionResult().observe(this, this::onPredictionResult);
        evaluationViewModel.getEvaluationResult().observe(this, this::onEvaluationResult);
        evaluationViewModel.getEvaluationLogs().observe(this, logs -> {
            evaluationLogAdapter.submitList(logs);
            binding.tvHistoryEmpty.setVisibility(logs == null || logs.isEmpty() ? View.VISIBLE : View.GONE);
        });
        evaluationViewModel.getLoading().observe(this, isLoading -> {
            boolean loading = isLoading != null && isLoading;
            binding.progressBar.setVisibility(loading ? View.VISIBLE : View.GONE);
            if (loading) {
                statePlaceholder.showLoading("正在处理中…");
            }
        });
        observeError(evaluationViewModel);
        observeOperationResult(evaluationViewModel, null);
    }

    private void showHazyImage(FileInfo fileInfo) {
        if (fileInfo == null || fileInfo.getUrl() == null) return;
        binding.ivHazy.setVisibility(View.VISIBLE);
        Glide.with(this).load(fileInfo.getUrl())
                .placeholder(R.drawable.ic_image)
                .error(R.drawable.ic_broken_image)
                .into(binding.ivHazy);
    }

    private void showClearImage(FileInfo fileInfo) {
        if (fileInfo == null || fileInfo.getUrl() == null) return;
        binding.ivClear.setVisibility(View.VISIBLE);
        Glide.with(this).load(fileInfo.getUrl())
                .placeholder(R.drawable.ic_image)
                .error(R.drawable.ic_broken_image)
                .into(binding.ivClear);
    }

    private void updateModelSpinner(List<AiModelVO> models) {
        modelOptions.clear();
        if (models != null) {
            modelOptions.addAll(models);
        }
        ViewUtils.updateModelSpinner(binding.spinnerModel, modelOptions);
    }

    private void onPredictionResult(ImageTaskResponse result) {
        if (result == null) return;
        statePlaceholder.hide();
        binding.btnEvaluate.setEnabled(true);
        if (result.getResultUrl() != null) {
            Glide.with(this).load(result.getResultUrl())
                    .placeholder(R.drawable.ic_image)
                    .error(R.drawable.ic_broken_image)
                    .into(binding.ivResult);
        }
    }

    private void onEvaluationResult(EvalResult result) {
        if (result == null) return;
        EvalTaskStatus status = result.getStatus();
        if (status == EvalTaskStatus.FAILED) {
            binding.tvStatus.setText("评估结论：失败 - " + (result.getErrorMessage() != null ? result.getErrorMessage() : ""));
            binding.tvStatus.setTextColor(getResources().getColor(android.R.color.holo_red_dark));
        } else if (status == EvalTaskStatus.COMPLETED) {
            binding.tvStatus.setText("评估结论：完成");
            binding.tvStatus.setTextColor(getResources().getColor(android.R.color.holo_green_dark));
        } else {
            binding.tvStatus.setText("评估结论：-");
            binding.tvStatus.setTextColor(getResources().getColor(android.R.color.darker_gray));
        }
        Map<String, Double> metrics = result.getMetrics();
        List<Map.Entry<String, Double>> entries = new ArrayList<>();
        if (metrics != null) {
            entries.addAll(metrics.entrySet());
        }
        metricAdapter.submitList(entries);
        binding.tvMetricsEmpty.setVisibility(entries.isEmpty() ? View.VISIBLE : View.GONE);
    }

    private void loadData() {
        evaluationViewModel.loadModels();
        evaluationViewModel.loadEvaluationLogs();
    }

    private void uploadHazyImage(Uri uri) {
        File tempFile = copyToCache(uri);
        if (tempFile != null) evaluationViewModel.uploadHazyImage(tempFile);
    }

    private void uploadClearImage(Uri uri) {
        File tempFile = copyToCache(uri);
        if (tempFile != null) evaluationViewModel.uploadClearImage(tempFile);
    }

    private File copyToCache(Uri uri) {
        File tempFile = UriUtils.copyToCache(this, uri);
        if (tempFile == null) {
            ToastUtils.showShort(this, "无法读取所选图片");
        }
        return tempFile;
    }
}
