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

import com.pei.dehaze.databinding.ActivityAlgorithmManageBinding;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmStatus;
import com.pei.dehaze.ui.system.viewmodel.AlgorithmManageViewModel;
import com.pei.dehaze.utils.ToastUtils;

import java.util.ArrayList;
import java.util.List;

public class AlgorithmManageActivity extends AppCompatActivity {

    private static final String[] STATUS_LABELS = {"全部", "草稿", "测试中", "待审核", "已发布", "已停用", "已归档"};
    private static final AlgorithmStatus[] STATUS_VALUES = {null,
            AlgorithmStatus.DRAFT, AlgorithmStatus.TESTING, AlgorithmStatus.PENDING_AUDIT,
            AlgorithmStatus.PUBLISHED, AlgorithmStatus.DISABLED, AlgorithmStatus.ARCHIVED};

    private AlgorithmManageViewModel viewModel;
    private ActivityAlgorithmManageBinding binding;
    private final List<Algorithm> flatList = new ArrayList<>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityAlgorithmManageBinding.inflate(getLayoutInflater());
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
                android.R.layout.simple_spinner_item, STATUS_LABELS);
        statusAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        binding.spinnerStatus.setAdapter(statusAdapter);

        binding.recyclerView.setLayoutManager(new LinearLayoutManager(this));
        binding.recyclerView.setAdapter(new AlgorithmListAdapter());

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
    }

    private void initViewModel() {
        viewModel = new ViewModelProvider(this).get(AlgorithmManageViewModel.class);
    }

    private void setupObservers() {
        viewModel.getAlgorithmList().observe(this, algorithms -> {
            flatList.clear();
            if (algorithms != null) flattenTree(algorithms, flatList);
            binding.recyclerView.getAdapter().notifyDataSetChanged();
            binding.tvEmpty.setVisibility(flatList.isEmpty() ? View.VISIBLE : View.GONE);
        });

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
        viewModel.loadAlgorithms();
    }

    private void flattenTree(List<Algorithm> tree, List<Algorithm> flat) {
        for (Algorithm a : tree) {
            flat.add(a);
            if (a.getChildren() != null && !a.getChildren().isEmpty()) {
                flattenTree(a.getChildren(), flat);
            }
        }
    }

    private class AlgorithmListAdapter extends RecyclerView.Adapter<AlgorithmListAdapter.ViewHolder> {

        @NonNull
        @Override
        public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
            View view = LayoutInflater.from(parent.getContext())
                    .inflate(android.R.layout.simple_list_item_2, parent, false);
            return new ViewHolder(view);
        }

        @Override
        public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
            Algorithm algo = flatList.get(position);
            holder.text1.setText(algo.getName());
            String status = algo.getStatus() != null ? algo.getStatus().getLabel() : "未知";
            String type = algo.getType() != null ? algo.getType() : "";
            holder.text2.setText(status + (type.isEmpty() ? "" : " | " + type));

            holder.itemView.setOnClickListener(v -> showActionDialog(algo));
        }

        @Override
        public int getItemCount() {
            return flatList.size();
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

    private void showActionDialog(Algorithm algo) {
        Long id = algo.getId();
        if (id == null) {
            ToastUtils.showShort(this, "无效的算法ID");
            return;
        }

        AlgorithmStatus status = algo.getStatus();
        boolean canPublish = status == AlgorithmStatus.PENDING_AUDIT || status == AlgorithmStatus.DISABLED;
        boolean canDisable = status == AlgorithmStatus.PUBLISHED;
        boolean canDelete = status == AlgorithmStatus.DRAFT || status == AlgorithmStatus.ARCHIVED
                || status == AlgorithmStatus.DISABLED;

        List<String> actions = new ArrayList<>();
        if (canPublish) actions.add(status == AlgorithmStatus.PENDING_AUDIT ? "审核通过(发布)" : "启用");
        if (canDisable) actions.add("停用");
        if (canDelete) actions.add("删除");

        if (actions.isEmpty()) {
            ToastUtils.showShort(this, "当前状态不支持操作");
            return;
        }

        new AlertDialog.Builder(this)
                .setTitle(algo.getName())
                .setItems(actions.toArray(new String[0]), (dialog, which) -> {
                    String action = actions.get(which);
                    if (action.contains("审核通过") || action.equals("启用")) {
                        viewModel.updateStatus(id, AlgorithmStatus.PUBLISHED);
                    } else if (action.equals("停用")) {
                        viewModel.updateStatus(id, AlgorithmStatus.DISABLED);
                    } else if (action.equals("删除")) {
                        new AlertDialog.Builder(this)
                                .setTitle("确认删除")
                                .setMessage("确定删除算法「" + algo.getName() + "」吗？")
                                .setPositiveButton("确定", (d, w) -> viewModel.deleteAlgorithm(id))
                                .setNegativeButton("取消", null)
                                .show();
                    }
                })
                .setNegativeButton("取消", null)
                .show();
    }
}
