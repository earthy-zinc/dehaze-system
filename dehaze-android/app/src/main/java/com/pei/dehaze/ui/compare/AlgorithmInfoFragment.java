package com.pei.dehaze.ui.compare;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.bumptech.glide.Glide;
import com.pei.dehaze.R;
import com.pei.dehaze.databinding.FragmentAlgorithmInfoBinding;
import com.pei.dehaze.repository.RepositoryCallback;
import com.pei.dehaze.sdk.model.algorithm_select.AlgorithmDetailVO;
import com.pei.dehaze.ui.compare.viewmodel.CompareViewModel;
import com.pei.dehaze.utils.StatePlaceholder;

import java.util.ArrayList;
import java.util.List;

/**
 * 算法信息 Fragment：展示当前去雾算法的详情（参数量/FLOPs/评分/使用次数/样例图）。
 * 调 AlgorithmSelectAPI.getDetail(id) 获取。
 */
public class AlgorithmInfoFragment extends Fragment {

    private CompareViewModel compareViewModel;
    private FragmentAlgorithmInfoBinding binding;
    private StatePlaceholder statePlaceholder;
    private SampleImageAdapter sampleImageAdapter;

    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        binding = FragmentAlgorithmInfoBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        compareViewModel = new ViewModelProvider(requireActivity()).get(CompareViewModel.class);

        statePlaceholder = new StatePlaceholder(binding.statePlaceholder.getRoot());
        statePlaceholder.showEmpty("请先完成去雾处理", R.drawable.ic_algorithm);

        sampleImageAdapter = new SampleImageAdapter();
        binding.rvSampleImages.setLayoutManager(
                new LinearLayoutManager(requireContext(), LinearLayoutManager.HORIZONTAL, false));
        binding.rvSampleImages.setAdapter(sampleImageAdapter);

        // 去雾处理完成后，加载算法详情
        compareViewModel.getPredictionResult().observe(getViewLifecycleOwner(), result -> {
            if (result == null) return;
            Long algorithmId = compareViewModel.getCurrentAlgorithmId();
            if (algorithmId != null) {
                loadAlgorithmDetail(algorithmId);
            }
        });
        compareViewModel.getMultiPredictionResults().observe(getViewLifecycleOwner(), results -> {
            if (results == null || results.isEmpty()) return;
            Long algorithmId = compareViewModel.getCurrentAlgorithmId();
            if (algorithmId != null) {
                loadAlgorithmDetail(algorithmId);
            }
        });
        compareViewModel.getLoading().observe(getViewLifecycleOwner(), isLoading -> {
            if (isLoading != null && isLoading) {
                statePlaceholder.showLoading("加载算法信息…");
            }
        });
    }

    private void loadAlgorithmDetail(long algorithmId) {
        compareViewModel.loadAlgorithmDetail(algorithmId, new RepositoryCallback<AlgorithmDetailVO>() {
            @Override
            public void onSuccess(AlgorithmDetailVO detail) {
                if (detail == null) {
                    statePlaceholder.showEmpty("算法详情加载失败", R.drawable.ic_algorithm);
                    return;
                }
                showAlgorithmDetail(detail);
            }

            @Override
            public void onError(String errorMessage) {
                statePlaceholder.showEmpty("算法详情加载失败：" + errorMessage, R.drawable.ic_algorithm);
            }
        });
    }

    private void showAlgorithmDetail(AlgorithmDetailVO detail) {
        statePlaceholder.hide();

        binding.tvAlgorithmName.setText(detail.getName() != null ? detail.getName() : "");
        binding.tvAlgorithmType.setText("类型：" + (detail.getType() != null ? detail.getType() : "-"));
        binding.tvAlgorithmVersion.setText("版本：" + (detail.getVersion() != null ? detail.getVersion() : "-"));
        binding.tvAlgorithmDescription.setText(detail.getDescription() != null ? detail.getDescription() : "");

        binding.tvParams.setText("参数量：" + (detail.getParams() != null ? detail.getParams() : "-"));
        binding.tvFlops.setText("FLOPs：" + (detail.getFlops() != null ? detail.getFlops() : "-"));
        binding.tvSize.setText("模型大小：" + (detail.getSize() != null ? detail.getSize() : "-"));

        binding.tvAvgRating.setText("平均评分：" + (detail.getAvgRating() != null
                ? String.format("%.2f", detail.getAvgRating()) : "-"));
        binding.tvRatingCount.setText("评价数：" + (detail.getRatingCount() != null ? detail.getRatingCount() : 0));
        binding.tvUsageCount.setText("使用次数：" + (detail.getUsageCount() != null ? detail.getUsageCount() : 0));

        // 样例效果图
        if (detail.getSampleImages() != null && !detail.getSampleImages().isEmpty()) {
            binding.tvSampleImagesTitle.setVisibility(View.VISIBLE);
            binding.rvSampleImages.setVisibility(View.VISIBLE);
            sampleImageAdapter.submitList(detail.getSampleImages());
        } else {
            binding.tvSampleImagesTitle.setVisibility(View.GONE);
            binding.rvSampleImages.setVisibility(View.GONE);
        }
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }

    /** 样例效果图 Adapter */
    private static class SampleImageAdapter extends RecyclerView.Adapter<SampleImageAdapter.VH> {
        private List<String> urls = new ArrayList<>();

        void submitList(List<String> newUrls) {
            this.urls = newUrls != null ? newUrls : new ArrayList<>();
            notifyDataSetChanged();
        }

        @NonNull
        @Override
        public VH onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
            ImageView iv = new ImageView(parent.getContext());
            int size = (int) (160 * parent.getResources().getDisplayMetrics().density);
            iv.setLayoutParams(new ViewGroup.LayoutParams(size, size));
            iv.setScaleType(ImageView.ScaleType.CENTER_CROP);
            int pad = (int) (4 * parent.getResources().getDisplayMetrics().density);
            iv.setPadding(pad, pad, pad, pad);
            return new VH(iv);
        }

        @Override
        public void onBindViewHolder(@NonNull VH holder, int position) {
            String url = urls.get(position);
            Glide.with(holder.imageView).load(url)
                    .placeholder(R.drawable.ic_image)
                    .error(R.drawable.ic_broken_image)
                    .into(holder.imageView);
        }

        @Override
        public int getItemCount() {
            return urls.size();
        }

        static class VH extends RecyclerView.ViewHolder {
            final ImageView imageView;
            VH(ImageView iv) {
                super(iv);
                this.imageView = iv;
            }
        }
    }
}
