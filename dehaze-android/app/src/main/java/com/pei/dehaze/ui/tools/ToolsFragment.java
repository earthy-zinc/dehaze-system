package com.pei.dehaze.ui.tools;

import android.content.Intent;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.inputmethod.EditorInfo;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;
import androidx.navigation.NavController;
import androidx.navigation.Navigation;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.pei.dehaze.R;
import com.pei.dehaze.databinding.FragmentToolsBinding;
import com.pei.dehaze.ui.algorithm.AlgorithmFragment;
import com.pei.dehaze.ui.algorithm_select.AlgorithmSelectActivity;
import com.pei.dehaze.ui.batch.BatchActivity;
import com.pei.dehaze.ui.input.InputHistoryActivity;
import com.pei.dehaze.utils.ToastUtils;

public class ToolsFragment extends Fragment {

    private ToolsViewModel toolsViewModel;
    private FragmentToolsBinding binding;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        binding = FragmentToolsBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        toolsViewModel = new ViewModelProvider(this).get(ToolsViewModel.class);

        initViews();
        setupObservers();
    }

    private void initViews() {
        // 搜索栏：对接算法搜索 API，结果跳转到 AlgorithmFragment 展示
        binding.etSearch.setOnEditorActionListener((v, actionId, event) -> {
            if (actionId == EditorInfo.IME_ACTION_SEARCH) {
                String keyword = binding.etSearch.getText() == null ? "" : binding.etSearch.getText().toString().trim();
                if (!keyword.isEmpty()) {
                    toolsViewModel.search(keyword);
                }
                return true;
            }
            return false;
        });

        // 快捷入口横滑区
        QuickEntryAdapter quickAdapter = new QuickEntryAdapter(entry -> {
            switch (entry.getId()) {
                case 1: // 处理历史
                    startActivity(new Intent(getActivity(), InputHistoryActivity.class));
                    break;
                case 2: // 我的收藏 -> profileFragment（后续 dev-personal 细化到收藏子页）
                    navigateTo(R.id.profileFragment);
                    break;
                case 3: // 批量处理 -> BatchActivity
                    startActivity(new Intent(getActivity(), BatchActivity.class));
                    break;
                case 4: // 算法选择 -> AlgorithmSelectActivity
                    startActivity(new Intent(getActivity(), AlgorithmSelectActivity.class));
                    break;
                default:
                    break;
            }
        });
        binding.rvQuickEntries.setLayoutManager(
                new LinearLayoutManager(requireContext(), RecyclerView.HORIZONTAL, false));
        binding.rvQuickEntries.setAdapter(quickAdapter);

        // 功能网格
        FeatureGridAdapter featureAdapter = new FeatureGridAdapter(item -> {
            switch (item.getAction()) {
                case "upload":
                    startActivity(new Intent(getActivity(), InputHistoryActivity.class));
                    break;
                case "algorithm_list":
                    startActivity(new Intent(getActivity(), AlgorithmSelectActivity.class));
                    break;
                case "dataset":
                    navigateTo(R.id.datasetFragment);
                    break;
                case "batch":
                    startActivity(new Intent(getActivity(), BatchActivity.class));
                    break;
                case "metrics":
                    navigateTo(R.id.action_global_evaluationActivity);
                    break;
                case "api_doc":
                    // API文档：暂无独立页面，跳转到算法列表作为参考入口
                    startActivity(new Intent(getActivity(), AlgorithmSelectActivity.class));
                    ToastUtils.showShort(getContext(), "API文档请查看算法详情页的接口说明");
                    break;
                default:
                    break;
            }
        });
        binding.rvFeatureGrid.setLayoutManager(new GridLayoutManager(requireContext(), 3));
        binding.rvFeatureGrid.setAdapter(featureAdapter);
    }

    private void navigateTo(int destinationId) {
        NavController navController = Navigation.findNavController(requireActivity(),
                R.id.nav_host_fragment_content_main);
        navController.navigate(destinationId);
    }

    private void setupObservers() {
        toolsViewModel.getQuickEntries().observe(getViewLifecycleOwner(), entries -> {
            RecyclerView.Adapter<?> adapter = binding.rvQuickEntries.getAdapter();
            if (adapter instanceof QuickEntryAdapter) {
                ((QuickEntryAdapter) adapter).submitList(entries);
            }
        });

        toolsViewModel.getFeatureItems().observe(getViewLifecycleOwner(), items -> {
            RecyclerView.Adapter<?> adapter = binding.rvFeatureGrid.getAdapter();
            if (adapter instanceof FeatureGridAdapter) {
                ((FeatureGridAdapter) adapter).submitList(items);
            }
        });

        // 搜索结果监听：跳转到 AlgorithmFragment 展示
        toolsViewModel.getSearchResults().observe(getViewLifecycleOwner(), results -> {
            if (results != null && !results.isEmpty()) {
                // 将搜索结果带入 AlgorithmFragment 展示
                Intent intent = new Intent(getActivity(), AlgorithmSelectActivity.class);
                intent.putExtra(AlgorithmSelectActivity.EXTRA_SEARCH_KEYWORD,
                        binding.etSearch.getText() != null ? binding.etSearch.getText().toString().trim() : "");
                startActivity(intent);
            }
        });

        toolsViewModel.getError().observe(getViewLifecycleOwner(), errorMessage -> {
            if (errorMessage != null && !errorMessage.isEmpty()) {
                ToastUtils.showShort(getContext(), errorMessage);
                toolsViewModel.clearError();
            }
        });
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}
