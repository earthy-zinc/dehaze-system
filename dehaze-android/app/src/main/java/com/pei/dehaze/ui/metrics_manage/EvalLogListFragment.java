package com.pei.dehaze.ui.metrics_manage;

import android.app.AlertDialog;
import android.graphics.Typeface;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;

import com.pei.dehaze.R;
import com.pei.dehaze.databinding.FragmentEvalLogListBinding;
import com.pei.dehaze.sdk.model.evaluation.EvaluationLogVO;
import com.pei.dehaze.ui.metrics_manage.adapter.EvalLogAdapter;
import com.pei.dehaze.ui.metrics_manage.viewmodel.MetricsManageViewModel;
import com.pei.dehaze.utils.ToastUtils;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * 评估日志列表 Fragment（支持对比模式）
 */
public class EvalLogListFragment extends Fragment {

    private MetricsManageViewModel viewModel;
    private EvalLogAdapter adapter;
    private FragmentEvalLogListBinding binding;
    private Long filterAlgorithmId;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        binding = FragmentEvalLogListBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        viewModel = new ViewModelProvider(requireActivity()).get(MetricsManageViewModel.class);

        adapter = new EvalLogAdapter();
        binding.recyclerView.setLayoutManager(new LinearLayoutManager(getContext()));
        binding.recyclerView.setAdapter(adapter);

        binding.swipeRefresh.setOnRefreshListener(this::loadData);

        binding.btnCompare.setOnClickListener(v -> showCompareDialog());
        binding.btnCancelCompare.setOnClickListener(v -> {
            adapter.setCompareMode(false);
            binding.layoutCompareBar.setVisibility(View.GONE);
        });

        adapter.setSelectionListener(selectedIds ->
                binding.tvSelectedCount.setText("已选 " + selectedIds.size() + " 条"));

        setupObservers();
    }

    public void setFilterAlgorithmId(Long id) {
        this.filterAlgorithmId = id;
        loadData();
    }

    public void toggleCompareMode() {
        boolean newMode = !adapter.isCompareMode();
        adapter.setCompareMode(newMode);
        binding.layoutCompareBar.setVisibility(newMode ? View.VISIBLE : View.GONE);
    }

    private void loadData() {
        viewModel.loadEvalLogs(filterAlgorithmId);
    }

    private void setupObservers() {
        viewModel.getEvalLogs().observe(getViewLifecycleOwner(), list -> {
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

    private void showCompareDialog() {
        Set<Long> selectedIds = adapter.getSelectedIds();
        if (selectedIds.size() < 2) {
            ToastUtils.showShort(getContext(), "请至少选择2条记录进行对比");
            return;
        }

        // 从当前列表中查找选中的记录
        List<EvaluationLogVO> currentList = viewModel.getEvalLogs().getValue();
        if (currentList == null) return;
        List<EvaluationLogVO> selected = new ArrayList<>();
        for (EvaluationLogVO log : currentList) {
            if (log.getId() != null && selectedIds.contains(log.getId())) {
                selected.add(log);
            }
        }

        View dialogView = LayoutInflater.from(requireContext())
                .inflate(R.layout.item_metrics_compare, null);
        LinearLayout tableLayout = dialogView.findViewById(R.id.layout_table);

        // 构建对比表格
        buildCompareTable(tableLayout, selected);

        new AlertDialog.Builder(requireContext())
                .setTitle("指标对比")
                .setView(dialogView)
                .setPositiveButton("关闭", null)
                .show();
    }

    private void buildCompareTable(LinearLayout container, List<EvaluationLogVO> logs) {
        container.removeAllViews();

        // 表头
        LinearLayout headerRow = new LinearLayout(getContext());
        headerRow.setOrientation(LinearLayout.HORIZONTAL);
        headerRow.addView(createCell("记录", true));

        String[] metricNames = {"PSNR", "SSIM", "LPIPS"};
        for (String name : metricNames) {
            headerRow.addView(createCell(name, true));
        }
        container.addView(headerRow);

        // 数据行
        for (EvaluationLogVO log : logs) {
            LinearLayout row = new LinearLayout(getContext());
            row.setOrientation(LinearLayout.HORIZONTAL);
            row.addView(createCell("#" + log.getId(), false));

            Map<String, Double> metrics = log.getResult() != null ? log.getResult().getMetrics() : null;
            for (String name : metricNames) {
                String value = "--";
                if (metrics != null && metrics.containsKey(name)) {
                    value = String.format("%.4f", metrics.get(name));
                }
                row.addView(createCell(value, false));
            }
            container.addView(row);
        }
    }

    private TextView createCell(String text, boolean isHeader) {
        TextView tv = new TextView(getContext());
        tv.setText(text);
        tv.setTextSize(12);
        tv.setPadding(8, 6, 8, 6);
        tv.setTextColor(isHeader ? 0xFFFFFFFF : 0xFF333333);
        if (isHeader) {
            tv.setBackgroundColor(0xFF1976D2);
            tv.setTypeface(Typeface.defaultFromStyle(Typeface.BOLD));
        }
        tv.setLayoutParams(new LinearLayout.LayoutParams(
                0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        return tv;
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}
