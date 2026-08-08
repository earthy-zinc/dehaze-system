package com.pei.dehaze.ui.system;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.pei.dehaze.databinding.ActivityTaskManageBinding;
import com.pei.dehaze.sdk.model.task.TaskVO;
import com.pei.dehaze.ui.system.viewmodel.TaskManageViewModel;
import com.pei.dehaze.utils.ToastUtils;

import java.util.List;

public class TaskManageActivity extends AppCompatActivity {

    private TaskManageViewModel viewModel;
    private ActivityTaskManageBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityTaskManageBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());
        initViews();
        initViewModel();
        setupObservers();
        loadData();
    }

    private void initViews() {
        setSupportActionBar(binding.toolbar);
        binding.toolbar.setNavigationOnClickListener(v -> finish());

        ArrayAdapter<String> statusAdapter = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_item,
                new String[]{"全部", "等待中", "处理中", "成功", "失败", "已取消"});
        statusAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        binding.spinnerStatus.setAdapter(statusAdapter);

        binding.recyclerView.setLayoutManager(new LinearLayoutManager(this));
        binding.recyclerView.setAdapter(new TaskListAdapter());

        binding.swipeRefresh.setOnRefreshListener(this::loadData);

        binding.btnSearch.setOnClickListener(v -> {
            viewModel.setKeywords(binding.etKeywords.getText().toString().trim());
            loadData();
        });

        binding.btnReset.setOnClickListener(v -> {
            binding.etKeywords.setText("");
            binding.spinnerStatus.setSelection(0);
            viewModel.resetQuery();
            loadData();
        });

        binding.btnPrev.setOnClickListener(v -> viewModel.prevPage());
        binding.btnNext.setOnClickListener(v -> viewModel.nextPage());
    }

    private void initViewModel() {
        viewModel = new ViewModelProvider(this).get(TaskManageViewModel.class);
    }

    private void setupObservers() {
        viewModel.getTaskList().observe(this, tasks -> {
            binding.recyclerView.getAdapter().notifyDataSetChanged();
            binding.tvEmpty.setVisibility(tasks == null || tasks.isEmpty() ? View.VISIBLE : View.GONE);
            updatePageInfo();
        });

        viewModel.getTotal().observe(this, total -> updatePageInfo());

        viewModel.getLoading().observe(this, isLoading ->
                binding.swipeRefresh.setRefreshing(isLoading != null && isLoading));

        viewModel.getError().observe(this, errorMessage -> {
            if (errorMessage != null && !errorMessage.isEmpty()) {
                ToastUtils.showShort(this, errorMessage);
                viewModel.clearError();
            }
        });

        viewModel.getOperationResult().observe(this, result -> {
            if (result != null && !result.isEmpty()) {
                ToastUtils.showShort(this, result);
                viewModel.clearOperationResult();
            }
        });
    }

    private void loadData() {
        viewModel.loadTasks();
    }

    private void updatePageInfo() {
        long total = viewModel.getTotal().getValue() != null ? viewModel.getTotal().getValue() : 0L;
        int pageNum = viewModel.getPageNum();
        int pageSize = viewModel.getPageSize();
        int totalPages = Math.max(1, (int) Math.ceil(total * 1.0 / pageSize));
        binding.tvPageInfo.setText("第 " + pageNum + " 页 / 共 " + totalPages + " 页 (共 " + total + " 条)");
    }

    private class TaskListAdapter extends RecyclerView.Adapter<TaskListAdapter.ViewHolder> {
        @NonNull
        @Override
        public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
            View view = LayoutInflater.from(parent.getContext())
                    .inflate(android.R.layout.simple_list_item_2, parent, false);
            return new ViewHolder(view);
        }

        @Override
        public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
            List<TaskVO> list = viewModel.getTaskList().getValue();
            if (list == null || position >= list.size()) return;
            TaskVO task = list.get(position);
            holder.text1.setText("任务#" + task.getTaskId().substring(0, Math.min(8, task.getTaskId().length())));
            String status = task.getStatus() != null ? task.getStatus().getLabel() : "未知";
            String info = status + " | 进度:" + task.getProgress() + "%";
            holder.text2.setText(info);

            holder.itemView.setOnLongClickListener(v -> {
                showActionDialog(task);
                return true;
            });
        }

        @Override
        public int getItemCount() {
            List<TaskVO> list = viewModel.getTaskList().getValue();
            return list != null ? list.size() : 0;
        }

        class ViewHolder extends RecyclerView.ViewHolder {
            TextView text1, text2;
            ViewHolder(View itemView) {
                super(itemView);
                text1 = itemView.findViewById(android.R.id.text1);
                text2 = itemView.findViewById(android.R.id.text2);
            }
        }
    }

    private void showActionDialog(TaskVO task) {
        String status = task.getStatus() != null ? task.getStatus().getValue() : "";

        java.util.List<String> actions = new java.util.ArrayList<>();
        if ("WAITING".equals(status) || "PROCESSING".equals(status)) {
            actions.add("取消任务");
        }

        if (actions.isEmpty()) {
            ToastUtils.showShort(this, "当前状态不支持操作");
            return;
        }

        new AlertDialog.Builder(this)
                .setTitle("任务#" + task.getTaskId().substring(0, Math.min(8, task.getTaskId().length())))
                .setItems(actions.toArray(new String[0]), (dialog, which) -> {
                    if ("取消任务".equals(actions.get(which))) {
                        new AlertDialog.Builder(this)
                                .setTitle("确认取消")
                                .setMessage("确定取消该任务吗？")
                                .setPositiveButton("确定", (d, w) -> viewModel.cancelTask(task.getTaskId()))
                                .setNegativeButton("取消", null)
                                .show();
                    }
                })
                .setNegativeButton("取消", null)
                .show();
    }
}
