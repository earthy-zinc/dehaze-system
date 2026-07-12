package com.pei.dehaze.ui.evaluation;

import android.net.Uri;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.Spinner;
import android.widget.TextView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;

import com.bumptech.glide.Glide;
import com.google.android.material.button.MaterialButton;
import com.google.android.material.card.MaterialCardView;
import com.pei.dehaze.R;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.evaluation.EvalResult;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.sdk.model.prediction.PredResult;
import com.pei.dehaze.ui.evaluation.adapter.EvaluationLogAdapter;
import com.pei.dehaze.ui.evaluation.adapter.MetricAdapter;
import com.pei.dehaze.ui.evaluation.viewmodel.EvaluationViewModel;
import com.pei.dehaze.utils.ToastUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class EvaluationActivity extends AppCompatActivity {

    private EvaluationViewModel evaluationViewModel;

    private Toolbar toolbar;
    private Spinner spinnerAlgorithm;
    private ImageView ivHazy;
    private ImageView ivClear;
    private MaterialButton btnPredict;
    private MaterialButton btnEvaluate;
    private ProgressBar progressBar;
    private MaterialCardView cardEvaluationResult;
    private TextView tvQualified;
    private androidx.recyclerview.widget.RecyclerView rvMetrics;
    private androidx.recyclerview.widget.RecyclerView rvHistory;

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
        setContentView(R.layout.activity_evaluation);

        initViews();
        initViewModel();
        setupObservers();
        loadData();
    }

    private void initViews() {
        toolbar = findViewById(R.id.toolbar);
        spinnerAlgorithm = findViewById(R.id.spinner_algorithm);
        ivHazy = findViewById(R.id.iv_hazy);
        ivClear = findViewById(R.id.iv_clear);
        btnPredict = findViewById(R.id.btn_predict);
        btnEvaluate = findViewById(R.id.btn_evaluate);
        progressBar = findViewById(R.id.progress_bar);
        cardEvaluationResult = findViewById(R.id.card_evaluation_result);
        tvQualified = findViewById(R.id.tv_qualified);
        rvMetrics = findViewById(R.id.rv_metrics);
        rvHistory = findViewById(R.id.rv_history);

        setSupportActionBar(toolbar);
        toolbar.setNavigationOnClickListener(v -> finish());

        findViewById(R.id.btn_select_hazy).setOnClickListener(v -> pickHazyLauncher.launch("image/*"));
        findViewById(R.id.btn_select_clear).setOnClickListener(v -> pickClearLauncher.launch("image/*"));

        btnPredict.setOnClickListener(v -> onPredictClick());
        btnEvaluate.setOnClickListener(v -> onEvaluateClick());

        metricAdapter = new MetricAdapter();
        rvMetrics.setLayoutManager(new LinearLayoutManager(this));
        rvMetrics.setAdapter(metricAdapter);

        evaluationLogAdapter = new EvaluationLogAdapter();
        rvHistory.setLayoutManager(new LinearLayoutManager(this));
        rvHistory.setAdapter(evaluationLogAdapter);
    }

    private void onPredictClick() {
        Long algorithmId = getCurrentAlgorithmId();
        if (algorithmId == null) {
            ToastUtils.showShort(this, "请先选择算法");
            return;
        }
        evaluationViewModel.predict(algorithmId);
    }

    private void onEvaluateClick() {
        Long algorithmId = getCurrentAlgorithmId();
        if (algorithmId == null) {
            ToastUtils.showShort(this, "请先选择算法");
            return;
        }
        evaluationViewModel.evaluate(algorithmId);
    }

    private Long getCurrentAlgorithmId() {
        int pos = spinnerAlgorithm.getSelectedItemPosition();
        if (pos < 0 || pos >= algorithmOptions.size()) return null;
        Option option = algorithmOptions.get(pos);
        return safeParseLong(option.getValue());
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
        evaluationViewModel.getEvaluationLogs().observe(this, logs ->
                evaluationLogAdapter.submitList(logs));
        evaluationViewModel.getLoading().observe(this, isLoading ->
                progressBar.setVisibility(isLoading != null && isLoading ? View.VISIBLE : View.GONE));
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
        ivHazy.setVisibility(View.VISIBLE);
        Glide.with(this).load(fileInfo.getUrl())
                .placeholder(R.drawable.ic_image)
                .error(R.drawable.ic_broken_image)
                .into(ivHazy);
    }

    private void showClearImage(FileInfo fileInfo) {
        if (fileInfo == null || fileInfo.getUrl() == null) return;
        ivClear.setVisibility(View.VISIBLE);
        Glide.with(this).load(fileInfo.getUrl())
                .placeholder(R.drawable.ic_image)
                .error(R.drawable.ic_broken_image)
                .into(ivClear);
    }

    private void updateAlgorithmSpinner(List<Option> options) {
        algorithmOptions.clear();
        if (options != null) {
            algorithmOptions.addAll(options);
        }
        List<String> labels = new ArrayList<>();
        for (Option opt : algorithmOptions) {
            labels.add(opt.getLabel());
        }
        ArrayAdapter<String> adapter = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_item, labels);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinnerAlgorithm.setAdapter(adapter);
    }

    private void onPredictionResult(PredResult result) {
        if (result == null) return;
        btnEvaluate.setEnabled(true);
    }

    private void onEvaluationResult(EvalResult result) {
        if (result == null) return;
        cardEvaluationResult.setVisibility(View.VISIBLE);
        Boolean qualified = result.getQualified();
        if (qualified != null) {
            if (qualified) {
                tvQualified.setText("评估结论：合格");
                tvQualified.setTextColor(getResources().getColor(android.R.color.holo_green_dark));
            } else {
                tvQualified.setText("评估结论：不合格");
                tvQualified.setTextColor(getResources().getColor(android.R.color.holo_red_dark));
            }
        } else {
            tvQualified.setText("评估结论：-");
            tvQualified.setTextColor(getResources().getColor(android.R.color.darker_gray));
        }
        Map<String, Double> metrics = result.getMetrics();
        List<Map.Entry<String, Double>> entries = new ArrayList<>();
        if (metrics != null) {
            entries.addAll(metrics.entrySet());
        }
        metricAdapter.submitList(entries);
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
        try {
            InputStream is = getContentResolver().openInputStream(uri);
            if (is == null) {
                ToastUtils.showShort(this, "无法读取所选图片");
                return null;
            }
            String fileName = getFileNameFromUri(uri);
            File tempFile = new File(getCacheDir(), fileName != null ? fileName : "upload_temp");
            try (FileOutputStream fos = new FileOutputStream(tempFile)) {
                byte[] buffer = new byte[4096];
                int len;
                while ((len = is.read(buffer)) != -1) {
                    fos.write(buffer, 0, len);
                }
            }
            is.close();
            return tempFile;
        } catch (Exception e) {
            ToastUtils.showShort(this, "读取图片失败: " + e.getMessage());
            return null;
        }
    }

    private String getFileNameFromUri(Uri uri) {
        String result = null;
        if ("content".equals(uri.getScheme())) {
            try (android.database.Cursor cursor = getContentResolver().query(uri, null, null, null, null)) {
                if (cursor != null && cursor.moveToFirst()) {
                    int idx = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME);
                    if (idx >= 0) {
                        result = cursor.getString(idx);
                    }
                }
            }
        }
        if (result == null) {
            result = uri.getLastPathSegment();
        }
        return result;
    }

    private Long safeParseLong(String value) {
        if (value == null) return null;
        try {
            return Long.parseLong(value);
        } catch (NumberFormatException e) {
            return null;
        }
    }
}
