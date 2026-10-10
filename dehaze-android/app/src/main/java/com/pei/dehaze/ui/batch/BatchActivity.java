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

import com.google.gson.Gson;
import com.google.gson.JsonSyntaxException;
import com.pei.dehaze.databinding.ActivityBatchBinding;
import com.pei.dehaze.sdk.model.ai_model.AiModelVO;
import com.pei.dehaze.sdk.model.image_task.DehazeParams;
import com.pei.dehaze.ui.batch.adapter.BatchImageAdapter;
import com.pei.dehaze.ui.batch.adapter.BatchResultAdapter;
import com.pei.dehaze.ui.batch.model.BatchImageItem;
import com.pei.dehaze.ui.batch.viewmodel.BatchViewModel;
import com.pei.dehaze.ui.common.adapter.ModelSelectAdapter;
import com.pei.dehaze.utils.ToastUtils;
import com.pei.dehaze.utils.UriUtils;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * 批量处理 Activity（L2）
 * 步骤1: 选择图片 → 步骤2: 选择模型 → 步骤3: 逐张提交图像任务 → 查看结果
 */
public class BatchActivity extends BaseActivity {

    private BatchViewModel viewModel;
    private ActivityBatchBinding binding;

    private BatchImageAdapter imageAdapter;
    private ModelSelectAdapter modelAdapter;
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
        viewModel.loadOptions();
    }

    private void initViewModels() {
        viewModel = new ViewModelProvider(this).get(BatchViewModel.class);
    }

    private void initLaunchers() {
        cameraLauncher = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(),
                result -> {
                    if (result.getResultCode() == RESULT_OK && pendingCameraUri != null) {
                        addImage(pendingCameraUri);
                    }
                });

        imagePickerLauncher = registerForActivityResult(
                new ActivityResultContracts.GetContent(),
                uri -> {
                    if (uri != null) {
                        addImage(uri);
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
            if (!checkCapacity()) return;
            dispatchCameraIntent();
        });

        binding.btnPickImages.setOnClickListener(v -> {
            if (!checkCapacity()) return;
            imagePickerLauncher.launch("image/*");
        });

        // 模型选择
        modelAdapter = new ModelSelectAdapter(model -> {
            viewModel.setSelectedModel(model);
            binding.tvModelHint.setText("已选择: "
                    + (model.getDisplayName() != null ? model.getDisplayName() : model.getModelId()));
        });
        binding.rvModels.setLayoutManager(new LinearLayoutManager(this));
        binding.rvModels.setAdapter(modelAdapter);

        // 结果列表
        resultAdapter = new BatchResultAdapter();
        binding.rvResults.setLayoutManager(new LinearLayoutManager(this));
        binding.rvResults.setAdapter(resultAdapter);

        List<BatchImageItem> results = viewModel.getResultItems().getValue();
        resultAdapter.setOnResultActionListener(new BatchResultAdapter.OnResultActionListener() {
            @Override
            public void onViewResult(BatchImageItem item) {
                if (item.getResultUrl() != null) {
                    startActivity(new Intent(Intent.ACTION_VIEW, Uri.parse(item.getResultUrl())));
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
            if (viewModel.getSelectedModel() == null) {
                ToastUtils.showShort(this, "请先选择模型");
                return;
            }
            if (Boolean.TRUE.equals(viewModel.getIsProcessing().getValue())) {
                ToastUtils.showShort(this, "正在处理中...");
                return;
            }
            DehazeParams params = parseParams();
            if (params == null) return;
            viewModel.setParams(params);
            viewModel.startBatchProcessing();
        });
    }

    /** 参数留空取默认值；JSON 非法或超出范围时提示并返回 null */
    private DehazeParams parseParams() {
        String text = binding.etParams.getText() == null
                ? "" : binding.etParams.getText().toString().trim();
        if (text.isEmpty()) return new DehazeParams();
        DehazeParams params;
        try {
            params = new Gson().fromJson(text, DehazeParams.class);
        } catch (JsonSyntaxException e) {
            params = null;
        }
        if (params == null) {
            ToastUtils.showShort(this, "模型参数不是合法 JSON");
            return null;
        }
        String invalid = params.validate();
        if (invalid != null) {
            ToastUtils.showShort(this, invalid);
            return null;
        }
        return params;
    }

    private void addImage(Uri uri) {
        File file = UriUtils.copyToCache(this, uri);
        if (file == null) {
            ToastUtils.showShort(this, "无法读取所选图片");
            return;
        }
        viewModel.addImage(uri, file);
    }

    private boolean checkCapacity() {
        if (viewModel.canAddMore()) return true;
        Integer limit = viewModel.getMaxImages().getValue();
        ToastUtils.showShort(this, "最多选择" + (limit != null ? limit : 0) + "张图片");
        return false;
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
            Integer limit = viewModel.getMaxImages().getValue();
            binding.tvImageCount.setText("已选择 " + items.size() + "/" + (limit != null ? limit : 0) + " 张");
        });

        viewModel.getMaxImages().observe(this, limit -> binding.tvLimitHint.setText(
                "最多" + (limit != null ? limit : 0) + "张（会员等级权益），支持拍照或相册选择"));

        viewModel.getModels().observe(this, list -> {
            modelAdapter.submitList(list);
            AiModelVO selected = viewModel.getSelectedModel();
            modelAdapter.setSelectedModelId(selected != null ? selected.getModelId() : null);
            if (selected != null) {
                binding.tvModelHint.setText("已选择: " + (selected.getDisplayName() != null
                        ? selected.getDisplayName() : selected.getModelId()));
            }
        });

        viewModel.getResultItems().observe(this, list -> {
            resultAdapter.submitList(list);
            binding.tvResultEmpty.setVisibility(list == null || list.isEmpty() ? View.VISIBLE : View.GONE);
        });

        viewModel.getIsProcessing().observe(this, processing -> {
            binding.btnStartBatch.setEnabled(!processing);
            binding.progressBar.setVisibility(processing ? View.VISIBLE : View.GONE);
        });

        viewModel.getProgressCount().observe(this, count ->
                binding.tvProgress.setText(count != null && !count.isEmpty() ? "进度: " + count : ""));

        observeError(viewModel);
    }
}
