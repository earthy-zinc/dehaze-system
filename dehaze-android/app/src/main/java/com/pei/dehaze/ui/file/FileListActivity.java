package com.pei.dehaze.ui.file;

import android.Manifest;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import com.pei.dehaze.utils.ToastUtils;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import com.pei.dehaze.ui.common.BaseActivity;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import android.widget.ProgressBar;

import com.pei.dehaze.R;
import com.pei.dehaze.databinding.ActivityFileListBinding;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.ui.file.adapter.FileAdapter;
import com.pei.dehaze.ui.file.viewmodel.FileViewModel;
import com.pei.dehaze.utils.UriUtils;
import com.pei.dehaze.utils.ViewUtils;

import java.io.File;
import java.util.List;

/**
 * 文件列表页
 * 支持分页查询、关键字搜索、上传、下载、删除、查看详情
 */
public class FileListActivity extends BaseActivity {

    private FileViewModel fileViewModel;
    private FileAdapter fileAdapter;
    private ActivityFileListBinding binding;

    private final ActivityResultLauncher<String> pickFileLauncher =
            registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
                if (uri != null) {
                    uploadFile(uri);
                }
            });

    private final ActivityResultLauncher<String> storagePermissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), granted -> {
                if (granted) {
                    ToastUtils.showShort(this, "存储权限已授予");
                } else {
                    ToastUtils.showShort(this, "需要存储权限才能下载文件");
                }
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityFileListBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        initViews();
        initViewModel();
        setupObservers();
        loadData();
    }

    private void initViews() {
        if (getSupportActionBar() != null) {
            getSupportActionBar().setDisplayHomeAsUpEnabled(true);
            getSupportActionBar().setTitle("文件管理");
        }

        fileAdapter = new FileAdapter();
        binding.recyclerView.setLayoutManager(new LinearLayoutManager(this));
        binding.recyclerView.setAdapter(fileAdapter);

        binding.recyclerView.addOnScrollListener(new RecyclerView.OnScrollListener() {
            @Override
            public void onScrolled(@NonNull RecyclerView rv, int dx, int dy) {
                LinearLayoutManager lm = (LinearLayoutManager) rv.getLayoutManager();
                if (lm != null) {
                    int totalItemCount = lm.getItemCount();
                    int lastVisible = lm.findLastVisibleItemPosition();
                    if (lastVisible + 1 >= totalItemCount) {
                        fileViewModel.loadMore();
                    }
                }
            }
        });

        fileAdapter.setOnFileClickListener(new FileAdapter.OnFileClickListener() {
            @Override
            public void onFileClick(FileInfo file) {
                fileViewModel.getFileDetail(file.getId());
            }

            @Override
            public void onDownloadClick(FileInfo file) {
                if (checkStoragePermission()) {
                    confirmDownload(file);
                } else {
                    storagePermissionLauncher.launch(Manifest.permission.WRITE_EXTERNAL_STORAGE);
                }
            }

            @Override
            public void onDeleteClick(FileInfo file) {
                confirmDelete(file);
            }
        });

        binding.btnSearch.setOnClickListener(v -> {
            String keywords = binding.etSearch.getText().toString().trim();
            fileViewModel.searchFiles(keywords);
        });

        binding.btnUpload.setOnClickListener(v -> {
            pickFileLauncher.launch("*/*");
        });

        binding.swipeRefresh.setOnRefreshListener(() -> fileViewModel.loadFiles());
    }

    private void initViewModel() {
        fileViewModel = new ViewModelProvider(this).get(FileViewModel.class);
    }

    private void setupObservers() {
        fileViewModel.getFileList().observe(this, files -> fileAdapter.submitList(files));

        fileViewModel.getLoading().observe(this, isLoading -> {
            binding.swipeRefresh.setRefreshing(isLoading != null && isLoading);
            binding.progressBar.setVisibility(isLoading != null && isLoading ? View.VISIBLE : View.GONE);
        });

        observeError(fileViewModel);

        observeOperationResult(fileViewModel, null);

        fileViewModel.getFileDetail().observe(this, file -> showDetailDialog(file));
    }

    private void loadData() {
        fileViewModel.loadFiles();
    }

    private boolean checkStoragePermission() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                == PackageManager.PERMISSION_GRANTED;
    }

    private void confirmDownload(FileInfo file) {
        new AlertDialog.Builder(this)
                .setTitle("下载文件")
                .setMessage("确认下载文件 \"" + file.getName() + "\" 吗？")
                .setPositiveButton("下载", (d, w) -> fileViewModel.downloadFile(file))
                .setNegativeButton("取消", null)
                .show();
    }

    private void confirmDelete(FileInfo file) {
        new AlertDialog.Builder(this)
                .setTitle("删除文件")
                .setMessage("确认删除该文件吗？删除后不可恢复。")
                .setPositiveButton("删除", (d, w) -> fileViewModel.deleteFile(file.getId()))
                .setNegativeButton("取消", null)
                .show();
    }

    private void showDetailDialog(FileInfo file) {
        View view = LayoutInflater.from(this).inflate(R.layout.dialog_file_detail, null);
        bindDetail(view, file);
        new AlertDialog.Builder(this)
                .setTitle("文件详情")
                .setView(view)
                .setPositiveButton("关闭", null)
                .show();
    }

    private void bindDetail(View view, FileInfo file) {
        ViewUtils.setText(view, R.id.tv_detail_id, String.valueOf(file.getId()));
        ViewUtils.setText(view, R.id.tv_detail_name, file.getName());
        ViewUtils.setText(view, R.id.tv_detail_type, file.getType());
        ViewUtils.setText(view, R.id.tv_detail_size, file.getSize());
        ViewUtils.setText(view, R.id.tv_detail_path, file.getPath());
        ViewUtils.setText(view, R.id.tv_detail_object_name, file.getObjectName());
        ViewUtils.setText(view, R.id.tv_detail_md5, file.getMd5());
        ViewUtils.setText(view, R.id.tv_detail_url, file.getUrl());
        ViewUtils.setText(view, R.id.tv_detail_create_time, file.getCreateTime());
    }

    private void uploadFile(Uri uri) {
        File tempFile = UriUtils.copyToCache(this, uri);
        if (tempFile == null) {
            ToastUtils.showShort(this, "无法读取所选文件");
            return;
        }
        fileViewModel.uploadFile(tempFile);
    }
}
