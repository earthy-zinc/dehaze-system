package com.pei.dehaze.ui.dataset;

import android.os.Bundle;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.RadioGroup;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;
import androidx.navigation.Navigation;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import androidx.swiperefreshlayout.widget.SwipeRefreshLayout;

import com.google.android.material.button.MaterialButton;
import com.google.android.material.textfield.TextInputEditText;
import com.pei.dehaze.R;
import com.pei.dehaze.repository.DatasetRepository;
import com.pei.dehaze.sdk.model.dataset.Dataset;
import com.pei.dehaze.utils.ToastUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * 数据集列表页（树形列表 + CRUD + 懒加载子节点）
 */
public class DatasetFragment extends Fragment {

    private DatasetViewModel viewModel;
    private DatasetTreeAdapter adapter;
    private SwipeRefreshLayout swipeRefresh;
    private EditText etKeywords;
    private MaterialButton btnAdd;
    private MaterialButton btnBatchDelete;
    private MaterialButton btnCancelSelect;
    private MaterialButton btnSelectAll;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        return inflater.inflate(R.layout.fragment_dataset, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        viewModel = new ViewModelProvider(this).get(DatasetViewModel.class);

        etKeywords = view.findViewById(R.id.et_keywords);
        swipeRefresh = view.findViewById(R.id.swipe_refresh);
        RecyclerView recyclerView = view.findViewById(R.id.recycler_view);
        btnAdd = view.findViewById(R.id.btn_add);
        btnBatchDelete = view.findViewById(R.id.btn_batch_delete);
        btnCancelSelect = view.findViewById(R.id.btn_cancel_select);
        btnSelectAll = view.findViewById(R.id.btn_select_all);

        adapter = new DatasetTreeAdapter();
        adapter.setActionListener(new DatasetTreeAdapter.OnDatasetActionListener() {
            @Override
            public void onView(Dataset dataset) {
                navigateToDetail(dataset);
            }

            @Override
            public void onEdit(Dataset dataset) {
                showFormDialog(dataset, false);
            }

            @Override
            public void onDelete(Dataset dataset) {
                confirmDelete(dataset);
            }

            @Override
            public void onAddChild(Dataset parent) {
                showFormDialog(parent, true);
            }

            @Override
            public void onLazyLoadChildren(Dataset parent) {
                viewModel.loadChildren(parent.getId(), new DatasetRepositoryCallback(parent.getId()));
            }
        });
        adapter.setSelectionListener(selectedIds ->
                btnBatchDelete.setText(selectedIds.isEmpty() ? "批量删除" : "删除选中(" + selectedIds.size() + ")"));

        recyclerView.setLayoutManager(new LinearLayoutManager(getContext()));
        recyclerView.setAdapter(adapter);

        swipeRefresh.setOnRefreshListener(() -> viewModel.loadRoots());

        view.findViewById(R.id.btn_search).setOnClickListener(v -> {
            String kw = etKeywords.getText().toString().trim();
            if (kw.isEmpty()) {
                ToastUtils.showShort(getContext(), "请输入搜索关键词");
                return;
            }
            viewModel.search(kw);
        });

        view.findViewById(R.id.btn_reset).setOnClickListener(v -> {
            etKeywords.setText("");
            viewModel.clearSearch();
        });

        btnAdd.setOnClickListener(v -> showFormDialog(null, false));

        btnBatchDelete.setOnClickListener(v -> {
            if (!adapter.isSelectionMode()) {
                adapter.setSelectionMode(true);
                updateSelectionUI(true);
                ToastUtils.showShort(getContext(), "请勾选要删除的数据集");
            } else {
                confirmBatchDelete();
            }
        });

        btnCancelSelect.setOnClickListener(v -> {
            adapter.clearSelection();
            adapter.setSelectionMode(false);
            updateSelectionUI(false);
        });

        btnSelectAll.setOnClickListener(v -> adapter.selectAll());

        setupObservers();
        viewModel.loadRoots();
    }

    private void updateSelectionUI(boolean selectionMode) {
        btnCancelSelect.setVisibility(selectionMode ? View.VISIBLE : View.GONE);
        btnSelectAll.setVisibility(selectionMode ? View.VISIBLE : View.GONE);
        btnAdd.setVisibility(selectionMode ? View.GONE : View.VISIBLE);
        if (selectionMode) {
            btnBatchDelete.setText("删除选中");
        } else {
            btnBatchDelete.setText("批量删除");
        }
    }

    private void setupObservers() {
        viewModel.getRootDatasets().observe(getViewLifecycleOwner(), datasets -> {
            adapter.setRoots(datasets);
        });

        viewModel.getSearchResults().observe(getViewLifecycleOwner(), datasets -> {
            adapter.setSearchResults(datasets);
        });

        viewModel.getLoading().observe(getViewLifecycleOwner(), isLoading ->
                swipeRefresh.setRefreshing(isLoading != null && isLoading));

        viewModel.getError().observe(getViewLifecycleOwner(), errorMessage -> {
            if (errorMessage != null && !errorMessage.isEmpty()) {
                ToastUtils.showShort(getContext(), errorMessage);
                viewModel.clearError();
            }
        });

        viewModel.getOperationResult().observe(getViewLifecycleOwner(), result -> {
            if (result != null && !result.isEmpty()) {
                ToastUtils.showShort(getContext(), result);
                viewModel.clearOperationResult();
                if (result.startsWith("删除") && adapter.isSelectionMode()) {
                    adapter.clearSelection();
                    adapter.setSelectionMode(false);
                    updateSelectionUI(false);
                }
            }
        });
    }

    private void navigateToDetail(Dataset dataset) {
        if (dataset.getId() == null) return;
        Bundle args = new Bundle();
        args.putLong("dataset_id", dataset.getId());
        Navigation.findNavController(requireView())
                .navigate(R.id.action_datasetFragment_to_datasetDetailFragment, args);
    }

    private void confirmDelete(Dataset dataset) {
        new AlertDialog.Builder(requireContext())
                .setTitle("删除确认")
                .setMessage("确认删除数据集「" + safe(dataset.getName()) + "」吗？此操作将同时删除所有子数据集和图片数据，且不可恢复！")
                .setPositiveButton("确定", (dialog, which) -> viewModel.deleteDataset(dataset.getId()))
                .setNegativeButton("取消", null)
                .show();
    }

    private void confirmBatchDelete() {
        Set<Long> selectedIds = adapter.getSelectedIds();
        if (selectedIds.isEmpty()) {
            ToastUtils.showShort(getContext(), "请先选择要删除的数据集");
            return;
        }
        new AlertDialog.Builder(requireContext())
                .setTitle("批量删除确认")
                .setMessage("确认删除选中的 " + selectedIds.size() + " 个数据集吗？删除后不可恢复。")
                .setPositiveButton("确定", (dialog, which) ->
                        viewModel.batchDeleteDatasets(new ArrayList<>(selectedIds)))
                .setNegativeButton("取消", null)
                .show();
    }

    private void showFormDialog(Dataset existing, boolean isAddChild) {
        boolean isEdit = existing != null && !isAddChild;
        View formView = LayoutInflater.from(requireContext()).inflate(R.layout.dialog_dataset_form, null);

        TextInputEditText etName = formView.findViewById(R.id.et_name);
        TextInputEditText etParentId = formView.findViewById(R.id.et_parent_id);
        TextInputEditText etType = formView.findViewById(R.id.et_type);
        TextInputEditText etPath = formView.findViewById(R.id.et_path);
        TextInputEditText etDescription = formView.findViewById(R.id.et_description);
        RadioGroup rgStatus = formView.findViewById(R.id.rg_status);

        if (isEdit) {
            etName.setText(safe(existing.getName()));
            etParentId.setText(existing.getParentId() != null ? String.valueOf(existing.getParentId()) : "0");
            etType.setText(safe(existing.getType()));
            etPath.setText(safe(existing.getPath()));
            etDescription.setText(safe(existing.getDescription()));
            if (existing.getStatus() != null && existing.getStatus() == 0) {
                rgStatus.check(R.id.rb_status_disable);
            } else {
                rgStatus.check(R.id.rb_status_enable);
            }
        } else if (isAddChild) {
            etParentId.setText(String.valueOf(existing.getId()));
            rgStatus.check(R.id.rb_status_enable);
        } else {
            etParentId.setText("0");
            rgStatus.check(R.id.rb_status_enable);
        }

        new AlertDialog.Builder(requireContext())
                .setTitle(isEdit ? "修改数据集" : (isAddChild ? "新增子数据集" : "新增数据集"))
                .setView(formView)
                .setPositiveButton("确定", (dialog, which) -> {
                    String name = getText(etName);
                    String type = getText(etType);
                    String path = getText(etPath);
                    String parentIdStr = getText(etParentId);
                    String description = getText(etDescription);
                    int status = rgStatus.getCheckedRadioButtonId() == R.id.rb_status_enable ? 1 : 0;

                    if (TextUtils.isEmpty(name)) {
                        ToastUtils.showShort(getContext(), "数据集名称不能为空");
                        return;
                    }
                    if (TextUtils.isEmpty(type)) {
                        ToastUtils.showShort(getContext(), "数据集类型不能为空");
                        return;
                    }
                    if (TextUtils.isEmpty(path)) {
                        ToastUtils.showShort(getContext(), "存储路径不能为空");
                        return;
                    }

                    Dataset form = new Dataset();
                    form.setName(name);
                    form.setType(type);
                    form.setPath(path);
                    form.setDescription(description);
                    form.setParentId(TextUtils.isEmpty(parentIdStr) ? 0L : Long.parseLong(parentIdStr));
                    form.setStatus(status);

                    if (isEdit) {
                        viewModel.updateDataset(existing.getId(), form);
                    } else {
                        viewModel.addDataset(form);
                    }
                })
                .setNegativeButton("取消", null)
                .show();
    }

    private String getText(TextInputEditText et) {
        return et.getText() != null ? et.getText().toString().trim() : "";
    }

    private String safe(String s) {
        return s == null ? "" : s;
    }

    /**
     * 懒加载子节点回调
     */
    private class DatasetRepositoryCallback implements DatasetRepository.Callback<List<Dataset>> {
        private final long parentId;

        DatasetRepositoryCallback(long parentId) {
            this.parentId = parentId;
        }

        @Override
        public void onSuccess(List<Dataset> data) {
            adapter.setChildren(parentId, data);
        }

        @Override
        public void onError(String errorMessage) {
            ToastUtils.showShort(getContext(), "加载子节点失败: " + errorMessage);
        }
    }
}
