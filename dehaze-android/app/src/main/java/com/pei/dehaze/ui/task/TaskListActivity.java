package com.pei.dehaze.ui.task;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.MenuItem;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import androidx.swiperefreshlayout.widget.SwipeRefreshLayout;

import com.google.android.material.textfield.MaterialAutoCompleteTextView;
import com.google.android.material.textfield.TextInputEditText;
import com.pei.dehaze.R;
import com.pei.dehaze.sdk.model.task.ExportOptions;
import com.pei.dehaze.sdk.model.task.TaskCreateForm;
import com.pei.dehaze.sdk.model.task.TaskStatus;
import com.pei.dehaze.sdk.model.task.TaskType;
import com.pei.dehaze.sdk.model.task.TaskVO;
import com.pei.dehaze.ui.task.adapter.TaskAdapter;
import com.pei.dehaze.ui.task.viewmodel.TaskViewModel;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * 任务列表页
 * 支持任务列表、状态/类型筛选、创建、取消、下载、删除、查看详情
 */
public class TaskListActivity extends AppCompatActivity {

    private TaskViewModel taskViewModel;
    private TaskAdapter taskAdapter;
    private RecyclerView recyclerView;
    private SwipeRefreshLayout swipeRefreshLayout;
    private ProgressBar progressBar;

    private MaterialAutoCompleteTextView spStatus;
    private MaterialAutoCompleteTextView spType;

    private String currentStatus;
    private String currentType;

    private final ActivityResultLauncher<String> storagePermissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), granted -> {
                if (!granted) {
                    Toast.makeText(this, "需要存储权限才能下载文件", Toast.LENGTH_SHORT).show();
                }
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_task_list);

        initViews();
        initViewModel();
        setupObservers();
        loadData();
    }

    private void initViews() {
        if (getSupportActionBar() != null) {
            getSupportActionBar().setDisplayHomeAsUpEnabled(true);
            getSupportActionBar().setTitle("任务管理");
        }

        recyclerView = findViewById(R.id.recycler_view);
        swipeRefreshLayout = findViewById(R.id.swipe_refresh);
        progressBar = findViewById(R.id.progress_bar);
        spStatus = findViewById(R.id.sp_status);
        spType = findViewById(R.id.sp_type);

        setupStatusFilter();
        setupTypeFilter();

        taskAdapter = new TaskAdapter();
        recyclerView.setLayoutManager(new LinearLayoutManager(this));
        recyclerView.setAdapter(taskAdapter);

        recyclerView.addOnScrollListener(new RecyclerView.OnScrollListener() {
            @Override
            public void onScrolled(@NonNull RecyclerView recyclerView, int dx, int dy) {
                LinearLayoutManager lm = (LinearLayoutManager) recyclerView.getLayoutManager();
                if (lm != null) {
                    int totalItemCount = lm.getItemCount();
                    int lastVisible = lm.findLastVisibleItemPosition();
                    if (lastVisible + 1 >= totalItemCount) {
                        taskViewModel.loadMore();
                    }
                }
            }
        });

        taskAdapter.setOnTaskClickListener(new TaskAdapter.OnTaskClickListener() {
            @Override
            public void onTaskClick(TaskVO task) {
                taskViewModel.getTaskDetail(task.getTaskId());
            }

            @Override
            public void onCancelClick(TaskVO task) {
                confirmCancel(task);
            }

            @Override
            public void onDownloadClick(TaskVO task) {
                if (checkStoragePermission()) {
                    confirmDownload(task);
                } else {
                    storagePermissionLauncher.launch(Manifest.permission.WRITE_EXTERNAL_STORAGE);
                }
            }
        });

        findViewById(R.id.btn_create).setOnClickListener(v -> showCreateDialog());

        swipeRefreshLayout.setOnRefreshListener(() -> taskViewModel.loadTasks());
    }

    private void setupStatusFilter() {
        List<String> statusLabels = new ArrayList<>();
        statusLabels.add("全部");
        for (TaskStatus status : TaskStatus.values()) {
            statusLabels.add(status.getLabel());
        }
        ArrayAdapter<String> adapter = new ArrayAdapter<>(this, android.R.layout.simple_list_item_1, statusLabels);
        spStatus.setAdapter(adapter);
        spStatus.setText("全部", false);
        spStatus.setOnItemClickListener((parent, view, position, id) -> {
            if (position == 0) {
                currentStatus = null;
            } else {
                currentStatus = TaskStatus.values()[position - 1].getValue();
            }
            taskViewModel.filterByStatus(currentStatus);
        });
    }

    private void setupTypeFilter() {
        List<String> typeLabels = new ArrayList<>();
        typeLabels.add("全部");
        for (TaskType type : TaskType.values()) {
            typeLabels.add(type.getLabel());
        }
        ArrayAdapter<String> adapter = new ArrayAdapter<>(this, android.R.layout.simple_list_item_1, typeLabels);
        spType.setAdapter(adapter);
        spType.setText("全部", false);
        spType.setOnItemClickListener((parent, view, position, id) -> {
            if (position == 0) {
                currentType = null;
            } else {
                currentType = TaskType.values()[position - 1].getValue();
            }
            taskViewModel.filterByType(currentType);
        });
    }

    private void initViewModel() {
        taskViewModel = new TaskViewModel();
    }

    private void setupObservers() {
        taskViewModel.getTaskList().observe(this, tasks -> taskAdapter.submitList(tasks));

        taskViewModel.getLoading().observe(this, isLoading -> {
            swipeRefreshLayout.setRefreshing(isLoading != null && isLoading);
            progressBar.setVisibility(isLoading != null && isLoading ? View.VISIBLE : View.GONE);
        });

        taskViewModel.getError().observe(this, msg -> {
            if (!TextUtils.isEmpty(msg)) {
                Toast.makeText(this, msg, Toast.LENGTH_SHORT).show();
            }
        });

        taskViewModel.getOperationResult().observe(this, msg -> {
            if (!TextUtils.isEmpty(msg)) {
                Toast.makeText(this, msg, Toast.LENGTH_SHORT).show();
            }
        });

        taskViewModel.getTaskDetail().observe(this, task -> showDetailDialog(task));
    }

    private void loadData() {
        taskViewModel.loadTasks();
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        if (item.getItemId() == android.R.id.home) {
            finish();
            return true;
        }
        return super.onOptionsItemSelected(item);
    }

    private boolean checkStoragePermission() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                == PackageManager.PERMISSION_GRANTED;
    }

    private void confirmCancel(TaskVO task) {
        new AlertDialog.Builder(this)
                .setTitle("取消任务")
                .setMessage("确认取消该任务吗？")
                .setPositiveButton("确认取消", (d, w) -> taskViewModel.cancelTask(task.getTaskId()))
                .setNegativeButton("返回", null)
                .show();
    }

    private void confirmDownload(TaskVO task) {
        new AlertDialog.Builder(this)
                .setTitle("下载任务结果")
                .setMessage("确认下载任务 \"" + task.getTaskId() + "\" 的结果文件吗？")
                .setPositiveButton("下载", (d, w) -> taskViewModel.downloadTaskFile(task.getTaskId()))
                .setNegativeButton("取消", null)
                .show();
    }

    private void showDetailDialog(TaskVO task) {
        View view = LayoutInflater.from(this).inflate(R.layout.dialog_task_detail, null);
        bindDetail(view, task);
        new AlertDialog.Builder(this)
                .setTitle("任务详情")
                .setView(view)
                .setPositiveButton("关闭", null)
                .show();
    }

    private void bindDetail(View view, TaskVO task) {
        setText(view, R.id.tv_detail_task_id, task.getTaskId());
        setText(view, R.id.tv_detail_type, task.getTaskType());
        setText(view, R.id.tv_detail_status, TaskStatus.fromValue(task.getStatus()).getLabel());
        setText(view, R.id.tv_detail_progress, task.getProgress() + "%");
        setText(view, R.id.tv_detail_files,
                (task.getProcessedFiles() != null ? task.getProcessedFiles() : 0) + " / " + task.getTotalFiles());
        setText(view, R.id.tv_detail_created_at, task.getCreatedAt());
        setText(view, R.id.tv_detail_started_at, task.getStartedAt());
        setText(view, R.id.tv_detail_completed_at, task.getCompletedAt());
        setText(view, R.id.tv_detail_expires_at, task.getExpiresAt());
        setText(view, R.id.tv_detail_download_url, task.getDownloadUrl());
        setText(view, R.id.tv_detail_error, task.getError());
    }

    private void setText(View root, int viewId, String text) {
        TextView tv = root.findViewById(viewId);
        if (tv != null) {
            tv.setText(text != null ? text : "—");
        }
    }

    private void showCreateDialog() {
        View view = LayoutInflater.from(this).inflate(R.layout.dialog_task_create, null);

        MaterialAutoCompleteTextView etType = view.findViewById(R.id.et_task_type);
        TextInputEditText etTargetId = view.findViewById(R.id.et_target_id);
        TextInputEditText etTargetIds = view.findViewById(R.id.et_target_ids);

        List<String> typeLabels = new ArrayList<>();
        for (TaskType type : TaskType.values()) {
            typeLabels.add(type.getLabel());
        }
        ArrayAdapter<String> typeAdapter = new ArrayAdapter<>(this, android.R.layout.simple_list_item_1, typeLabels);
        etType.setAdapter(typeAdapter);

        final String[] selectedType = {null};
        etType.setOnItemClickListener((parent, v, position, id) -> selectedType[0] = TaskType.values()[position].getValue());

        new AlertDialog.Builder(this)
                .setTitle("创建任务")
                .setView(view)
                .setPositiveButton("创建", (d, w) -> {
                    if (selectedType[0] == null) {
                        Toast.makeText(this, "请选择任务类型", Toast.LENGTH_SHORT).show();
                        return;
                    }
                    TaskCreateForm form = buildCreateForm(selectedType[0], etTargetId, etTargetIds);
                    if (form == null) {
                        return;
                    }
                    taskViewModel.createTask(form);
                })
                .setNegativeButton("取消", null)
                .show();
    }

    private TaskCreateForm buildCreateForm(String type, TextInputEditText etTargetId, TextInputEditText etTargetIds) {
        TaskCreateForm form = new TaskCreateForm();
        form.setType(type);

        String targetIdStr = etTargetId.getText() != null ? etTargetId.getText().toString().trim() : "";
        String targetIdsStr = etTargetIds.getText() != null ? etTargetIds.getText().toString().trim() : "";

        boolean isBatch = TaskType.BATCH_DOWNLOAD.getValue().equals(type) ||
                TaskType.CUSTOM_EXPORT.getValue().equals(type);

        if (isBatch) {
            if (TextUtils.isEmpty(targetIdsStr)) {
                Toast.makeText(this, "批量任务请填写目标ID列表", Toast.LENGTH_SHORT).show();
                return null;
            }
            List<Long> ids = new ArrayList<>();
            for (String s : targetIdsStr.split(",")) {
                String trimmed = s.trim();
                if (!trimmed.isEmpty()) {
                    try {
                        ids.add(Long.parseLong(trimmed));
                    } catch (NumberFormatException e) {
                        Toast.makeText(this, "目标ID格式错误: " + trimmed, Toast.LENGTH_SHORT).show();
                        return null;
                    }
                }
            }
            form.setTargetIds(ids);
        } else {
            if (TextUtils.isEmpty(targetIdStr)) {
                Toast.makeText(this, "请填写目标资源ID", Toast.LENGTH_SHORT).show();
                return null;
            }
            try {
                form.setTargetId(Long.parseLong(targetIdStr));
            } catch (NumberFormatException e) {
                Toast.makeText(this, "目标ID格式错误", Toast.LENGTH_SHORT).show();
                return null;
            }
        }

        ExportOptions options = new ExportOptions();
        form.setOptions(options);
        return form;
    }
}
