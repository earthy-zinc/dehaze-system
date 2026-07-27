package com.pei.dehaze.ui.algorithm.adapter;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import android.content.res.ColorStateList;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.google.android.material.chip.Chip;
import com.pei.dehaze.R;
import com.pei.dehaze.utils.StringUtils;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmStatus;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class AlgorithmAdapter extends RecyclerView.Adapter<AlgorithmAdapter.AlgorithmViewHolder> {

    public interface OnAlgorithmActionListener {
        void onView(Algorithm algorithm);
        void onEdit(Algorithm algorithm);
        void onDelete(Algorithm algorithm);
        void onToggleStatus(Algorithm algorithm);
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

    public void setOnAlgorithmActionListener(OnAlgorithmActionListener listener) {
        this.actionListener = listener;
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
        Map<Long, Boolean> expandStates = new HashMap<>();
        for (Node node : flatNodes) {
            expandStates.put(node.algorithm.getId(), node.expanded);
        }
        flatNodes.clear();
        for (Algorithm algorithm : rootAlgorithms) {
            flattenWithState(algorithm, 0, true, expandStates);
        }
        notifyDataSetChanged();
    }

    private void flattenWithState(Algorithm algorithm, int depth, boolean parentExpanded, Map<Long, Boolean> expandStates) {
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

        AlgorithmViewHolder(@NonNull View itemView) {
            super(itemView);
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
        }

        void bind(Node node) {
            Algorithm algorithm = node.algorithm;
            tvName.setText(StringUtils.safe(algorithm.getName()));
            tvType.setText(StringUtils.safe(algorithm.getType()));
            tvDescription.setText(StringUtils.safe(algorithm.getDescription()));
            tvParams.setText(StringUtils.safe(algorithm.getParams()));
            tvFlops.setText(StringUtils.safe(algorithm.getFlops()));

            AlgorithmStatus status = algorithm.getStatus() != null ? algorithm.getStatus() : AlgorithmStatus.DRAFT;
            chipStatus.setText(status.getLabel());
            chipStatus.setChipBackgroundColor(ColorStateList.valueOf(statusColor(status)));
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
        }

    }

    private int statusColor(AlgorithmStatus status) {
        if (status == null) return 0xFF9E9E9E;
        switch (status) {
            case DRAFT: return 0xFF9E9E9E;
            case TESTING: return 0xFFFF9800;
            case PENDING_AUDIT: return 0xFF2196F3;
            case PUBLISHED: return 0xFF4CAF50;
            case DISABLED: return 0xFFE53935;
            case ARCHIVED: return 0xFF607D8B;
            default: return 0xFF9E9E9E;
        }
    }
}
