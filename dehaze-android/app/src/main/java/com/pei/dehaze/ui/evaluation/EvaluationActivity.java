package com.pei.dehaze.ui.evaluation;

import androidx.appcompat.app.AlertDialog;
import android.net.Uri;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.View;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;

import com.bumptech.glide.Glide;
import com.pei.dehaze.R;
import com.pei.dehaze.databinding.ActivityEvaluationBinding;
import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.evaluation.EvalResult;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.sdk.model.prediction.DehazeParams;
import com.pei.dehaze.sdk.model.prediction.PredEvalTaskStatus;
import com.pei.dehaze.sdk.model.prediction.PredResult;
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

public class EvaluationActivity extends AppCompatActivity {

    private EvaluationViewModel evaluationViewModel;
    private ActivityEvaluationBinding binding;
    private StatePlaceholder statePlaceholder;

    private final List<Option> algorithmOptions = new ArrayList<>();
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
        Long algorithmId = getCurrentAlgorithmId();
        if (algorithmId == null) {
            ToastUtils.showShort(this, "请先选择算法");
            return;
        }
        new AlertDialog.Builder(this)
                .setTitle("确认处理")
                .setMessage("确认开始去雾处理？")
                .setPositiveButton("确定", (d, w) -> evaluationViewModel.predict(algorithmId, new DehazeParams()))
                .setNegativeButton("取消", null)
                .show();
    }

    private void onEvaluateClick() {
        Long algorithmId = getCurrentAlgorithmId();
        if (algorithmId == null) {
            ToastUtils.showShort(this, "请先选择算法");
            return;
        }
        new AlertDialog.Builder(this)
                .setTitle("确认评估")
                .setMessage("确认开始指标评估？")
                .setPositiveButton("确定", (d, w) -> evaluationViewModel.evaluate(algorithmId))
                .setNegativeButton("取消", null)
                .show();
    }

    private Long getCurrentAlgorithmId() {
        int pos = binding.spinnerAlgorithm.getSelectedItemPosition();
        if (pos < 0 || pos >= algorithmOptions.size()) return null;
        Option option = algorithmOptions.get(pos);
        return option.getValue() == null ? null : StringUtils.safeParseLong(option.getValue(), 0L);
    }

    private void initViewModel() {
        evaluationViewModel = new ViewModelProvider(this).get(EvaluationViewModel.class);
    }

    private void setupObservers() {
        evaluationViewModel.getHazyFile().observe(this, this::showHazyImage);
        evaluationViewModel.getClearFile().observe(this, this::showClearImage);
        evaluationViewModel.getAlgorithmOptions().observe(this, this::updateAlgorithmSpinner);
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
        evaluationViewModel.getError().observe(this, errorMessage -> {
            if (!TextUtils.isEmpty(errorMessage)) {
                ToastUtils.showShort(this, errorMessage);
                evaluationViewModel.clearError();
            }
        });
        evaluationViewModel.getOperationResult().observe(this, result -> {
            if (!TextUtils.isEmpty(result)) {
                ToastUtils.showShort(this, result);
                evaluationViewModel.clearOperationResult();
            }
        });
    }

    private void showHazyImage(FileInfo fileInfo) {
        if (fileInfo == null || fileInfo.getUrl() == null) return;
        binding.ivHazy.setVisibility(View.VISIBLE);
        Glide.with(this).load(DehazeSDK.getInstance().resolveUrl(fileInfo.getUrl()))
                .placeholder(R.drawable.ic_image)
                .error(R.drawable.ic_broken_image)
                .into(binding.ivHazy);
    }

    private void showClearImage(FileInfo fileInfo) {
        if (fileInfo == null || fileInfo.getUrl() == null) return;
        binding.ivClear.setVisibility(View.VISIBLE);
        Glide.with(this).load(DehazeSDK.getInstance().resolveUrl(fileInfo.getUrl()))
                .placeholder(R.drawable.ic_image)
                .error(R.drawable.ic_broken_image)
                .into(binding.ivClear);
    }

    private void updateAlgorithmSpinner(List<Option> options) {
        algorithmOptions.clear();
        if (options != null) {
            algorithmOptions.addAll(options);
        }
        ViewUtils.updateAlgorithmSpinner(binding.spinnerAlgorithm, algorithmOptions);
    }

    private void onPredictionResult(PredResult result) {
        if (result == null) return;
        statePlaceholder.hide();
        binding.btnEvaluate.setEnabled(true);
        if (result.getResultUrl() != null) {
            Glide.with(this).load(DehazeSDK.getInstance().resolveUrl(result.getResultUrl()))
                    .placeholder(R.drawable.ic_image)
                    .error(R.drawable.ic_broken_image)
                    .into(binding.ivResult);
        }
    }

    private void onEvaluationResult(EvalResult result) {
        if (result == null) return;
        PredEvalTaskStatus status = result.getStatus();
        if (status == PredEvalTaskStatus.FAILED) {
            binding.tvStatus.setText("评估结论：失败 - " + (result.getErrorMessage() != null ? result.getErrorMessage() : ""));
            binding.tvStatus.setTextColor(getResources().getColor(android.R.color.holo_red_dark));
        } else if (status == PredEvalTaskStatus.COMPLETED) {
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
        evaluationViewModel.loadAlgorithmOptions();
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
