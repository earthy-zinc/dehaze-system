package com.pei.dehaze.ui.system;

import android.os.Bundle;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.EditText;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.pei.dehaze.R;
import com.pei.dehaze.databinding.ActivityDatasetManageBinding;
import com.pei.dehaze.sdk.model.dataset.Dataset;
import com.pei.dehaze.ui.system.viewmodel.DatasetManageViewModel;
import com.pei.dehaze.utils.ToastUtils;

import java.util.List;

public class DatasetManageActivity extends AppCompatActivity {

    private DatasetManageViewModel viewModel;
    private ActivityDatasetManageBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityDatasetManageBinding.inflate(getLayoutInflater());
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
                android.R.layout.simple_spinner_item, new String[]{"全部", "启用", "禁用"});
        statusAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        binding.spinnerStatus.setAdapter(statusAdapter);

        binding.recyclerView.setLayoutManager(new LinearLayoutManager(this));
        binding.recyclerView.setAdapter(new DatasetListAdapter());

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

        binding.btnAdd.setOnClickListener(v -> showDatasetFormDialog(null));

        binding.btnPrev.setOnClickListener(v -> viewModel.prevPage());
        binding.btnNext.setOnClickListener(v -> viewModel.nextPage());
    }

    private void initViewModel() {
        viewModel = new ViewModelProvider(this).get(DatasetManageViewModel.class);
    }

    private void setupObservers() {
        viewModel.getDatasetList().observe(this, datasets -> {
            binding.recyclerView.getAdapter().notifyDataSetChanged();
            binding.tvEmpty.setVisibility(datasets == null || datasets.isEmpty() ? View.VISIBLE : View.GONE);
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
        viewModel.loadDatasets();
    }

    private void updatePageInfo() {
        long total = viewModel.getTotal().getValue() != null ? viewModel.getTotal().getValue() : 0L;
        int pageNum = viewModel.getPageNum();
        int pageSize = viewModel.getPageSize();
        int totalPages = Math.max(1, (int) Math.ceil(total * 1.0 / pageSize));
        binding.tvPageInfo.setText("第 " + pageNum + " 页 / 共 " + totalPages + " 页 (共 " + total + " 条)");
    }

    private void showDatasetFormDialog(Dataset existing) {
        boolean isEdit = existing != null && existing.getId() != null;
        View view = LayoutInflater.from(this).inflate(R.layout.dialog_dataset_form, null);
        EditText etName = view.findViewById(R.id.et_name);
        EditText etDesc = view.findViewById(R.id.et_description);

        if (isEdit) {
            etName.setText(existing.getName() != null ? existing.getName() : "");
            etDesc.setText(existing.getDescription() != null ? existing.getDescription() : "");
        }

        new AlertDialog.Builder(this)
                .setTitle(isEdit ? "修改数据集" : "新增数据集")
                .setView(view)
                .setPositiveButton("确定", (dialog, which) -> {
                    String name = etName.getText().toString().trim();
                    String desc = etDesc.getText().toString().trim();
                    if (TextUtils.isEmpty(name)) {
                        ToastUtils.showShort(this, "请输入数据集名称");
                        return;
                    }
                    Dataset data = new Dataset();
                    data.setName(name);
                    data.setDescription(desc);
                    data.setType("folder");
                    if (isEdit) {
                        viewModel.updateDataset(existing.getId(), data);
                    } else {
                        viewModel.addDataset(data);
                    }
                })
                .setNegativeButton("取消", null)
                .show();
    }

    private class DatasetListAdapter extends RecyclerView.Adapter<DatasetListAdapter.ViewHolder> {
        @NonNull
        @Override
        public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
            View view = LayoutInflater.from(parent.getContext())
                    .inflate(android.R.layout.simple_list_item_2, parent, false);
            return new ViewHolder(view);
        }

        @Override
        public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
            List<Dataset> list = viewModel.getDatasetList().getValue();
            if (list == null || position >= list.size()) return;
            Dataset ds = list.get(position);
            holder.text1.setText(ds.getName());
            holder.text2.setText(ds.getType() != null ? ds.getType() : "数据集");

            holder.itemView.setOnClickListener(v -> {
                new AlertDialog.Builder(DatasetManageActivity.this)
                        .setTitle(ds.getName())
                        .setItems(new String[]{"编辑", "删除"}, (dialog, which) -> {
                            if (which == 0) {
                                showDatasetFormDialog(ds);
                            } else {
                                new AlertDialog.Builder(DatasetManageActivity.this)
                                        .setTitle("确认删除")
                                        .setMessage("确定删除数据集「" + ds.getName() + "」吗？")
                                        .setPositiveButton("确定", (d, w) -> viewModel.deleteDataset(ds.getId()))
                                        .setNegativeButton("取消", null)
                                        .show();
                            }
                        })
                        .setNegativeButton("取消", null)
                        .show();
            });
        }

        @Override
        public int getItemCount() {
            List<Dataset> list = viewModel.getDatasetList().getValue();
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
}
