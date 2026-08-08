package com.pei.dehaze.ui.dataset;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.inputmethod.EditorInfo;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;
import androidx.navigation.Navigation;
import androidx.recyclerview.widget.LinearLayoutManager;

import com.pei.dehaze.R;
import com.pei.dehaze.databinding.FragmentDatasetBinding;
import com.pei.dehaze.sdk.model.dataset.Dataset;
import com.pei.dehaze.ui.dataset.adapter.DatasetBrowseAdapter;
import com.pei.dehaze.utils.ToastUtils;

/**
 * 数据集浏览版（L2）
 * 从工具网格进入，展示公开/共享数据集列表+详情浏览，无管理操作
 */
public class DatasetFragment extends Fragment {

    private DatasetBrowseViewModel viewModel;
    private DatasetBrowseAdapter adapter;
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

        viewModel = new ViewModelProvider(this).get(DatasetBrowseViewModel.class);

        adapter = new DatasetBrowseAdapter();
        adapter.setOnBrowseActionListener(dataset -> navigateToDetail(dataset));

        binding.recyclerView.setLayoutManager(new LinearLayoutManager(getContext()));
        binding.recyclerView.setAdapter(adapter);

        binding.swipeRefresh.setOnRefreshListener(() -> viewModel.loadTree());

        binding.etKeywords.setOnEditorActionListener((v, actionId, event) -> {
            if (actionId == EditorInfo.IME_ACTION_SEARCH) {
                doSearch();
                return true;
            }
            return false;
        });
        binding.btnSearch.setOnClickListener(v -> doSearch());

        binding.btnReset.setOnClickListener(v -> {
            binding.etKeywords.setText("");
            viewModel.loadTree();
        });

        setupObservers();
        viewModel.loadTree();
    }

    private void doSearch() {
        String keyword = binding.etKeywords.getText() == null ? "" : binding.etKeywords.getText().toString().trim();
        if (keyword.isEmpty()) {
            viewModel.loadTree();
        } else {
            viewModel.search(keyword);
        }
    }

    private void setupObservers() {
        viewModel.getDatasetList().observe(getViewLifecycleOwner(), list -> {
            adapter.submitList(list);
            binding.tvEmpty.setVisibility(list == null || list.isEmpty() ? View.VISIBLE : View.GONE);
        });

        viewModel.getLoading().observe(getViewLifecycleOwner(), isLoading ->
                binding.swipeRefresh.setRefreshing(isLoading != null && isLoading));

        viewModel.getError().observe(getViewLifecycleOwner(), errorMessage -> {
            if (errorMessage != null && !errorMessage.isEmpty()) {
                ToastUtils.showShort(getContext(), errorMessage);
                viewModel.clearError();
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

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}
