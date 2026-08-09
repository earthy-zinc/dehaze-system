package com.pei.dehaze.ui.batch;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.view.View;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import com.pei.dehaze.ui.common.BaseActivity;
import androidx.core.content.FileProvider;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.LinearLayoutManager;

import com.pei.dehaze.databinding.ActivityBatchBinding;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.ui.batch.adapter.BatchAlgorithmAdapter;
import com.pei.dehaze.ui.batch.adapter.BatchImageAdapter;
import com.pei.dehaze.ui.batch.adapter.BatchResultAdapter;
import com.pei.dehaze.ui.batch.model.BatchImageItem;
import com.pei.dehaze.ui.batch.viewmodel.BatchViewModel;
import com.pei.dehaze.utils.ToastUtils;
import com.pei.dehaze.utils.UriUtils;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * 批量处理 Activity（L2）
 * 步骤1: 选择图片 → 步骤2: 选择算法 → 步骤3: 批量处理 → 查看结果
 */
public class BatchActivity extends BaseActivity {

    private static final int MAX_IMAGES = 20;

    private BatchViewModel viewModel;
    private ActivityBatchBinding binding;

    private BatchImageAdapter imageAdapter;
    private BatchAlgorithmAdapter algorithmAdapter;
    private BatchResultAdapter resultAdapter;

    private Uri pendingCameraUri;
    private ActivityResultLauncher<Intent> cameraLauncher;
    private ActivityResultLauncher<String> imagePickerLauncher;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityBatchBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        initViewModels();
        initLaunchers();
        initViews();
        setupObservers();
        viewModel.loadAlgorithms();
    }

    private void initViewModels() {
        viewModel = new ViewModelProvider(this).get(BatchViewModel.class);
    }

    private void initLaunchers() {
        cameraLauncher = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(),
                result -> {
                    if (result.getResultCode() == RESULT_OK && pendingCameraUri != null) {
                        viewModel.addImage(pendingCameraUri);
                    }
                });

        imagePickerLauncher = registerForActivityResult(
                new ActivityResultContracts.GetContent(),
                uri -> {
                    if (uri != null) {
                        viewModel.addImage(uri);
                    }
                });
    }

    private void initViews() {
        binding.toolbar.setNavigationOnClickListener(v -> finish());

        // 图片网格
        imageAdapter = new BatchImageAdapter();
        binding.rvImages.setLayoutManager(new GridLayoutManager(this, 4));
        binding.rvImages.setAdapter(imageAdapter);
        imageAdapter.setOnRemoveListener(position -> viewModel.removeImage(position));

        binding.btnCamera.setOnClickListener(v -> {
            if (imageAdapter.getItems().size() >= MAX_IMAGES) {
                ToastUtils.showShort(this, "最多选择" + MAX_IMAGES + "张图片");
                return;
            }
            dispatchCameraIntent();
        });

        binding.btnPickImages.setOnClickListener(v -> {
            if (imageAdapter.getItems().size() >= MAX_IMAGES) {
                ToastUtils.showShort(this, "最多选择" + MAX_IMAGES + "张图片");
                return;
            }
            imagePickerLauncher.launch("image/*");
        });

        // 算法选择
        algorithmAdapter = new BatchAlgorithmAdapter();
        binding.rvAlgorithms.setLayoutManager(new LinearLayoutManager(this));
        binding.rvAlgorithms.setAdapter(algorithmAdapter);
        algorithmAdapter.setOnAlgorithmSelectListener(algo -> {
            viewModel.setSelectedAlgorithmId(algo.getId());
            binding.tvAlgorithmHint.setText("已选择: " + algo.getName());
        });

        // 结果列表
        resultAdapter = new BatchResultAdapter();
        binding.rvResults.setLayoutManager(new LinearLayoutManager(this));
        binding.rvResults.setAdapter(resultAdapter);

        List<BatchImageItem> results = viewModel.getResultItems().getValue();
        resultAdapter.setOnResultActionListener(new BatchResultAdapter.OnResultActionListener() {
            @Override
            public void onViewResult(BatchImageItem item) {
                if (item.getResultUrl() != null) {
                    Intent intent = new Intent(Intent.ACTION_VIEW, Uri.parse(item.getResultUrl()));
                    startActivity(intent);
                }
            }

            @Override
            public void onRetry(BatchImageItem item) {
                viewModel.retryItem(item, results != null ? results : new ArrayList<>());
            }
        });

        binding.btnStartBatch.setOnClickListener(v -> {
            if (imageAdapter.getItems().isEmpty()) {
                ToastUtils.showShort(this, "请先选择图片");
                return;
            }
            if (viewModel.getSelectedAlgorithmId() <= 0) {
                ToastUtils.showShort(this, "请先选择算法");
                return;
            }
            if (Boolean.TRUE.equals(viewModel.getIsProcessing().getValue())) {
                ToastUtils.showShort(this, "正在处理中...");
                return;
            }
            viewModel.startBatchProcessing();
        });
    }

    private void dispatchCameraIntent() {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
            try {
                File photoFile = createImageFile();
                if (photoFile != null) {
                    pendingCameraUri = FileProvider.getUriForFile(this,
                            getPackageName() + ".fileprovider", photoFile);
                    takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, pendingCameraUri);
                    cameraLauncher.launch(takePictureIntent);
                }
            } catch (IOException e) {
                ToastUtils.showShort(this, "创建临时文件失败");
            }
        }
    }

    private File createImageFile() throws IOException {
        String fileName = "batch_" + System.currentTimeMillis() + ".jpg";
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        return new File(storageDir, fileName);
    }

    private void setupObservers() {
        viewModel.getImageItems().observe(this, items -> {
            imageAdapter.clear();
            for (BatchImageItem item : items) {
                imageAdapter.addItem(item);
            }
            binding.tvImageCount.setText("已选择 " + items.size() + "/" + MAX_IMAGES + " 张");
        });

        viewModel.getAlgorithms().observe(this, list -> {
            algorithmAdapter.submitList(list);
        });

        viewModel.getResultItems().observe(this, list -> {
            resultAdapter.submitList(list);
            binding.tvResultEmpty.setVisibility(list == null || list.isEmpty() ? View.VISIBLE : View.GONE);
        });

        viewModel.getIsProcessing().observe(this, processing -> {
            binding.btnStartBatch.setEnabled(!processing);
            binding.progressBar.setVisibility(processing ? View.VISIBLE : View.GONE);
        });

        viewModel.getProgressCount().observe(this, count -> {
            binding.tvProgress.setText(count != null && !count.isEmpty() ? "进度: " + count : "");
        });

        observeError(viewModel);
    }
}
