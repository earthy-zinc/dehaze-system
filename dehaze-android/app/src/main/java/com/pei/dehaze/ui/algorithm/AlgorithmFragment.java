package com.pei.dehaze.ui.algorithm;

import android.content.Intent;
import android.os.Bundle;
import android.text.TextUtils;
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
import androidx.recyclerview.widget.RecyclerView;

import com.pei.dehaze.R;
import com.pei.dehaze.databinding.FragmentAlgorithmBinding;
import com.pei.dehaze.sdk.model.algorithm_select.AlgorithmSelectNodeVO;
import com.pei.dehaze.sdk.model.recommendation.RecommendedAlgorithmVO;
import com.pei.dehaze.ui.algorithm.adapter.AlgorithmBrowseAdapter;
import com.pei.dehaze.ui.algorithm.adapter.AlgorithmRecommendCardAdapter;
import com.pei.dehaze.ui.algorithm.viewmodel.AlgorithmBrowseViewModel;
import com.pei.dehaze.ui.algorithm_select.AlgorithmSelectActivity;
import com.pei.dehaze.utils.ToastUtils;

/**
 * 算法库浏览版（L2）
 * 从工具网格进入，展示已发布算法列表+智能推荐+「使用该算法」带入去雾流程
 */
public class AlgorithmFragment extends Fragment {

    private FragmentAlgorithmBinding binding;
    private AlgorithmBrowseViewModel viewModel;
    private AlgorithmBrowseAdapter browseAdapter;
    private AlgorithmRecommendCardAdapter recommendAdapter;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        binding = FragmentAlgorithmBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        viewModel = new ViewModelProvider(this).get(AlgorithmBrowseViewModel.class);

        initViews();
        setupObservers();
        loadData();
    }

    private void initViews() {
        // 搜索
        binding.etKeywords.setOnEditorActionListener((v, actionId, event) -> {
            if (actionId == EditorInfo.IME_ACTION_SEARCH) {
                doSearch();
                return true;
            }
            return false;
        });
        binding.btnSearch.setOnClickListener(v -> doSearch());

        // 算法列表
        browseAdapter = new AlgorithmBrowseAdapter();
        binding.recyclerView.setLayoutManager(new LinearLayoutManager(getContext()));
        binding.recyclerView.setAdapter(browseAdapter);

        binding.swipeRefresh.setOnRefreshListener(this::loadData);

        browseAdapter.setOnBrowseActionListener(new AlgorithmBrowseAdapter.OnBrowseActionListener() {
            @Override
            public void onViewDetail(AlgorithmSelectNodeVO algorithm) {
                if (algorithm.getId() != null) {
                    Intent intent = new Intent(getActivity(), AlgorithmDetailActivity.class);
                    intent.putExtra("algorithm_id", algorithm.getId());
                    startActivity(intent);
                }
            }

            @Override
            public void onUse(AlgorithmSelectNodeVO algorithm) {
                if (algorithm.getId() != null) {
                    navigateToDehazeWithAlgorithm(algorithm.getId(), algorithm.getName());
                }
            }
        });

        // 推荐列表
        recommendAdapter = new AlgorithmRecommendCardAdapter();
        binding.rvRecommend.setLayoutManager(
                new LinearLayoutManager(getContext(), LinearLayoutManager.HORIZONTAL, false));
        binding.rvRecommend.setAdapter(recommendAdapter);

        recommendAdapter.setOnRecommendActionListener(vo -> {
            if (vo.getAlgorithmId() != null) {
                navigateToDehazeWithAlgorithm(vo.getAlgorithmId(), vo.getAlgorithmName());
            }
        });

        binding.tvDismissRecommend.setOnClickListener(v -> dismissRecommend());
    }

    private void doSearch() {
        String keyword = binding.etKeywords.getText() == null ? "" : binding.etKeywords.getText().toString().trim();
        if (TextUtils.isEmpty(keyword)) {
            viewModel.loadAlgorithmTree();
        } else {
            viewModel.search(keyword);
        }
    }

    private void dismissRecommend() {
        binding.layoutRecommend.setVisibility(View.GONE);
        binding.rvRecommend.setVisibility(View.GONE);
    }

    private void setupObservers() {
        viewModel.getAlgorithmList().observe(getViewLifecycleOwner(), list -> {
            browseAdapter.submitList(list);
            binding.tvEmpty.setVisibility(list == null || list.isEmpty() ? View.VISIBLE : View.GONE);
        });

        viewModel.getRecommendations().observe(getViewLifecycleOwner(), list -> {
            boolean hasRecommend = list != null && !list.isEmpty();
            binding.layoutRecommend.setVisibility(hasRecommend ? View.VISIBLE : View.GONE);
            binding.rvRecommend.setVisibility(hasRecommend ? View.VISIBLE : View.GONE);
            if (hasRecommend) {
                recommendAdapter.submitList(list);
            }
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

    private void loadData() {
        viewModel.loadAlgorithmTree();
        // 推荐暂时基于空参数，后端可能返回热门推荐
        viewModel.loadRecommendations(null, null);
    }

    /**
     * 使用该算法 → 跳转 AlgorithmSelectActivity 并带入去雾流程
     */
    private void navigateToDehazeWithAlgorithm(long algorithmId, String algorithmName) {
        Intent intent = new Intent(getActivity(), AlgorithmSelectActivity.class);
        intent.putExtra(AlgorithmSelectActivity.EXTRA_ALGORITHM_ID, algorithmId);
        intent.putExtra(AlgorithmSelectActivity.EXTRA_ALGORITHM_NAME, algorithmName);
        startActivity(intent);
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}
