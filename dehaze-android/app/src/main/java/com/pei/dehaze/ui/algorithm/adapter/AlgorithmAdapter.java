package com.pei.dehaze.ui.algorithm.adapter;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.CheckBox;
import android.widget.ImageView;
import android.widget.TextView;

import android.content.res.ColorStateList;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.google.android.material.chip.Chip;
import com.pei.dehaze.R;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmStatus;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class AlgorithmAdapter extends RecyclerView.Adapter<AlgorithmAdapter.AlgorithmViewHolder> {

    public interface OnAlgorithmActionListener {
        void onView(Algorithm algorithm);
        void onEdit(Algorithm algorithm);
        void onDelete(Algorithm algorithm);
        void onToggleStatus(Algorithm algorithm);
        void onToggleFavorite(Algorithm algorithm);
    }

    public interface OnSelectionChangedListener {
        void onSelectionChanged(Set<Integer> selectedIds);
    }

    private static class Node {
        Algorithm algorithm;
        int depth;
        boolean expanded;
        boolean hasChildren;

        Node(Algorithm algorithm, int depth, boolean expanded, boolean hasChildren) {
            this.algorithm = algorithm;
            this.depth = depth;
            this.expanded = expanded;
            this.hasChildren = hasChildren;
        }
    }

    private final List<Node> flatNodes = new ArrayList<>();
    private final List<Algorithm> rootAlgorithms = new ArrayList<>();
    private OnAlgorithmActionListener actionListener;
    private OnSelectionChangedListener selectionListener;
    private boolean selectionMode = false;
    private final Set<Integer> selectedIds = new HashSet<>();

    public void setOnAlgorithmActionListener(OnAlgorithmActionListener listener) {
        this.actionListener = listener;
    }

    public void setOnSelectionChangedListener(OnSelectionChangedListener listener) {
        this.selectionListener = listener;
    }

    public void setData(List<Algorithm> algorithms) {
        rootAlgorithms.clear();
        if (algorithms != null) {
            rootAlgorithms.addAll(algorithms);
        }
        rebuildFlatNodes();
        notifyDataSetChanged();
    }

    private void rebuildFlatNodes() {
        flatNodes.clear();
        for (Algorithm algorithm : rootAlgorithms) {
            flatten(algorithm, 0, true);
        }
    }

    private void flatten(Algorithm algorithm, int depth, boolean parentExpanded) {
        if (!parentExpanded) {
            return;
        }
        boolean hasChildren = algorithm.getChildren() != null && !algorithm.getChildren().isEmpty();
        Node node = new Node(algorithm, depth, true, hasChildren);
        flatNodes.add(node);
        if (hasChildren && node.expanded) {
            for (Algorithm child : algorithm.getChildren()) {
                flatten(child, depth + 1, true);
            }
        }
    }

    private void rebuildAndNotify() {
        Map<Integer, Boolean> expandStates = new HashMap<>();
        for (Node node : flatNodes) {
            expandStates.put(node.algorithm.getId(), node.expanded);
        }
        flatNodes.clear();
        for (Algorithm algorithm : rootAlgorithms) {
            flattenWithState(algorithm, 0, true, expandStates);
        }
        notifyDataSetChanged();
    }

    private void flattenWithState(Algorithm algorithm, int depth, boolean parentExpanded, Map<Integer, Boolean> expandStates) {
        if (!parentExpanded) {
            return;
        }
        boolean hasChildren = algorithm.getChildren() != null && !algorithm.getChildren().isEmpty();
        boolean expanded = expandStates.getOrDefault(algorithm.getId(), true);
        Node node = new Node(algorithm, depth, expanded, hasChildren);
        flatNodes.add(node);
        if (hasChildren && expanded) {
            for (Algorithm child : algorithm.getChildren()) {
                flattenWithState(child, depth + 1, true, expandStates);
            }
        }
    }

    public void setSelectionMode(boolean selectionMode) {
        this.selectionMode = selectionMode;
        if (!selectionMode) {
            selectedIds.clear();
        }
        notifyItemRangeChanged(0, getItemCount());
    }

    public boolean isSelectionMode() {
        return selectionMode;
    }

    public void selectAll() {
        for (Node node : flatNodes) {
            selectedIds.add(node.algorithm.getId());
        }
        notifyItemRangeChanged(0, getItemCount());
        notifySelectionChanged();
    }

    public void clearSelection() {
        selectedIds.clear();
        notifyItemRangeChanged(0, getItemCount());
        notifySelectionChanged();
    }

    public Set<Integer> getSelectedIds() {
        return new HashSet<>(selectedIds);
    }

    public String getSelectedIdsString() {
        if (selectedIds.isEmpty()) {
            return "";
        }
        StringBuilder sb = new StringBuilder();
        for (Integer id : selectedIds) {
            if (sb.length() > 0) {
                sb.append(",");
            }
            sb.append(id);
        }
        return sb.toString();
    }

    private void notifySelectionChanged() {
        if (selectionListener != null) {
            selectionListener.onSelectionChanged(new HashSet<>(selectedIds));
        }
    }

    @Override
    public int getItemCount() {
        return flatNodes.size();
    }

    @NonNull
    @Override
    public AlgorithmViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_algorithm, parent, false);
        return new AlgorithmViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull AlgorithmViewHolder holder, int position) {
        holder.bind(flatNodes.get(position));
    }

    class AlgorithmViewHolder extends RecyclerView.ViewHolder {
        private CheckBox cbSelect;
        private TextView tvName;
        private TextView tvType;
        private TextView tvDescription;
        private TextView tvParams;
        private TextView tvFlops;
        private Chip chipStatus;
        private ImageView ivExpand;
        private View indentView;
        private TextView tvView;
        private TextView tvEdit;
        private TextView tvDelete;
        private TextView tvToggleStatus;
        private TextView tvFavorite;

        AlgorithmViewHolder(@NonNull View itemView) {
            super(itemView);
            cbSelect = itemView.findViewById(R.id.cb_select);
            tvName = itemView.findViewById(R.id.tv_algorithm_name);
            tvType = itemView.findViewById(R.id.tv_algorithm_type);
            tvDescription = itemView.findViewById(R.id.tv_algorithm_description);
            tvParams = itemView.findViewById(R.id.tv_algorithm_params);
            tvFlops = itemView.findViewById(R.id.tv_algorithm_flops);
            chipStatus = itemView.findViewById(R.id.chip_status);
            ivExpand = itemView.findViewById(R.id.iv_expand);
            indentView = itemView.findViewById(R.id.indent);
            tvView = itemView.findViewById(R.id.tv_view);
            tvEdit = itemView.findViewById(R.id.tv_edit);
            tvDelete = itemView.findViewById(R.id.tv_delete);
            tvToggleStatus = itemView.findViewById(R.id.tv_toggle_status);
            tvFavorite = itemView.findViewById(R.id.tv_favorite);
        }

        void bind(Node node) {
            Algorithm algorithm = node.algorithm;
            tvName.setText(safe(algorithm.getName()));
            tvType.setText(safe(algorithm.getType()));
            tvDescription.setText(safe(algorithm.getDescription()));
            tvParams.setText(safe(algorithm.getParams()));
            tvFlops.setText(safe(algorithm.getFlops()));

            // 状态 Chip
            int statusValue = algorithm.getStatus() != null ? algorithm.getStatus() : 0;
            AlgorithmStatus status = AlgorithmStatus.fromValue(statusValue);
            chipStatus.setText(status.getLabel());
            chipStatus.setChipBackgroundColor(ColorStateList.valueOf(statusColor(statusValue)));
            chipStatus.setTextColor(0xFFFFFFFF);

            int padding = node.depth * 32;
            indentView.getLayoutParams().width = padding;
            indentView.requestLayout();

            if (node.hasChildren) {
                ivExpand.setVisibility(View.VISIBLE);
                ivExpand.setImageResource(node.expanded ? R.drawable.ic_arrow_down : R.drawable.ic_arrow_right);
                ivExpand.setOnClickListener(v -> {
                    node.expanded = !node.expanded;
                    rebuildAndNotify();
                });
            } else {
                ivExpand.setVisibility(View.INVISIBLE);
                ivExpand.setOnClickListener(null);
            }

            if (selectionMode) {
                cbSelect.setVisibility(View.VISIBLE);
                cbSelect.setChecked(selectedIds.contains(algorithm.getId()));
                cbSelect.setOnCheckedChangeListener((button, checked) -> {
                    if (checked) {
                        selectedIds.add(algorithm.getId());
                    } else {
                        selectedIds.remove(algorithm.getId());
                    }
                    notifySelectionChanged();
                });
                hideActions();
            } else {
                cbSelect.setVisibility(View.GONE);
                cbSelect.setOnCheckedChangeListener(null);
                showActions();
            }

            tvView.setOnClickListener(v -> {
                if (actionListener != null) actionListener.onView(algorithm);
            });
            tvEdit.setOnClickListener(v -> {
                if (actionListener != null) actionListener.onEdit(algorithm);
            });
            tvDelete.setOnClickListener(v -> {
                if (actionListener != null) actionListener.onDelete(algorithm);
            });
            tvToggleStatus.setOnClickListener(v -> {
                if (actionListener != null) actionListener.onToggleStatus(algorithm);
            });
            tvFavorite.setOnClickListener(v -> {
                if (actionListener != null) actionListener.onToggleFavorite(algorithm);
            });

            itemView.setOnLongClickListener(v -> {
                if (!selectionMode) {
                    setSelectionMode(true);
                    selectedIds.add(algorithm.getId());
                    notifyItemRangeChanged(0, getItemCount());
                    notifySelectionChanged();
                    return true;
                }
                return false;
            });
        }

        private void showActions() {
            tvView.setVisibility(View.VISIBLE);
            tvEdit.setVisibility(View.VISIBLE);
            tvDelete.setVisibility(View.VISIBLE);
            tvToggleStatus.setVisibility(View.VISIBLE);
            tvFavorite.setVisibility(View.VISIBLE);
        }

        private void hideActions() {
            tvView.setVisibility(View.GONE);
            tvEdit.setVisibility(View.GONE);
            tvDelete.setVisibility(View.GONE);
            tvToggleStatus.setVisibility(View.GONE);
            tvFavorite.setVisibility(View.GONE);
        }

        private String safe(String s) {
            return s == null ? "" : s;
        }
    }

    private int statusColor(int status) {
        switch (status) {
            case 0: return 0xFF9E9E9E; // 草稿-灰
            case 1: return 0xFFFF9800; // 测试中-橙
            case 2: return 0xFF2196F3; // 待审核-蓝
            case 3: return 0xFF4CAF50; // 已发布-绿
            case 4: return 0xFFE53935; // 已停用-红
            case 5: return 0xFF607D8B; // 已归档-深灰
            default: return 0xFF9E9E9E;
        }
    }
}
