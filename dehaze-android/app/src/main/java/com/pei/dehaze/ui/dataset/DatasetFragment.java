package com.pei.dehaze.ui.dataset;

import android.os.Bundle;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.RadioGroup;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;
import androidx.navigation.Navigation;
import androidx.recyclerview.widget.LinearLayoutManager;

import com.google.android.material.textfield.TextInputEditText;
import com.pei.dehaze.R;
import com.pei.dehaze.databinding.FragmentDatasetBinding;
import com.pei.dehaze.repository.RepositoryCallback;
import com.pei.dehaze.sdk.model.dataset.Dataset;
import com.pei.dehaze.utils.StringUtils;
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
    private FragmentDatasetBinding binding;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        binding = FragmentDatasetBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        viewModel = new ViewModelProvider(this).get(DatasetViewModel.class);

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
                binding.btnBatchDelete.setText(selectedIds.isEmpty() ? "批量删除" : "删除选中(" + selectedIds.size() + ")"));

        binding.recyclerView.setLayoutManager(new LinearLayoutManager(getContext()));
        binding.recyclerView.setAdapter(adapter);

        binding.swipeRefresh.setOnRefreshListener(() -> viewModel.loadRoots());

        binding.btnSearch.setOnClickListener(v -> {
            String kw = binding.etKeywords.getText().toString().trim();
            if (kw.isEmpty()) {
                ToastUtils.showShort(getContext(), "请输入搜索关键词");
                return;
            }
            viewModel.search(kw);
        });

        binding.btnReset.setOnClickListener(v -> {
            binding.etKeywords.setText("");
            viewModel.clearSearch();
        });

        binding.btnAdd.setOnClickListener(v -> showFormDialog(null, false));

        binding.btnBatchDelete.setOnClickListener(v -> {
            if (!adapter.isSelectionMode()) {
                adapter.setSelectionMode(true);
                updateSelectionUI(true);
                ToastUtils.showShort(getContext(), "请勾选要删除的数据集");
            } else {
                confirmBatchDelete();
            }
        });

        binding.btnCancelSelect.setOnClickListener(v -> {
            adapter.clearSelection();
            adapter.setSelectionMode(false);
            updateSelectionUI(false);
        });

        binding.btnSelectAll.setOnClickListener(v -> adapter.selectAll());

        setupObservers();
        viewModel.loadRoots();
    }

    private void updateSelectionUI(boolean selectionMode) {
        binding.btnCancelSelect.setVisibility(selectionMode ? View.VISIBLE : View.GONE);
        binding.btnSelectAll.setVisibility(selectionMode ? View.VISIBLE : View.GONE);
        binding.btnAdd.setVisibility(selectionMode ? View.GONE : View.VISIBLE);
        if (selectionMode) {
            binding.btnBatchDelete.setText("删除选中");
        } else {
            binding.btnBatchDelete.setText("批量删除");
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
                binding.swipeRefresh.setRefreshing(isLoading != null && isLoading));

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
                .setMessage("确认删除数据集「" + StringUtils.safe(dataset.getName()) + "」吗？此操作将同时删除所有子数据集和图片数据，且不可恢复！")
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
            etName.setText(StringUtils.safe(existing.getName()));
            etParentId.setText(existing.getParentId() != null ? String.valueOf(existing.getParentId()) : "0");
            etType.setText(StringUtils.safe(existing.getType()));
            etPath.setText(StringUtils.safe(existing.getPath()));
            etDescription.setText(StringUtils.safe(existing.getDescription()));
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
                    String name = StringUtils.getText(etName);
                    String type = StringUtils.getText(etType);
                    String path = StringUtils.getText(etPath);
                    String parentIdStr = StringUtils.getText(etParentId);
                    String description = StringUtils.getText(etDescription);
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

    /**
     * 懒加载子节点回调
     */
    private class DatasetRepositoryCallback implements RepositoryCallback<List<Dataset>> {
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

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}
