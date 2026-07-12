package com.pei.dehaze.ui.system.adapter;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.pei.dehaze.R;
import com.pei.dehaze.sdk.model.dept.DeptVO;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class DeptAdapter extends RecyclerView.Adapter<DeptAdapter.DeptViewHolder> {

    public interface OnDeptActionListener {
        void onEdit(DeptVO dept);
        void onDelete(DeptVO dept);
        void onAddChild(DeptVO dept);
    }

    private static class Node {
        DeptVO dept;
        int depth;
        boolean expanded;
        boolean hasChildren;
        Node(DeptVO dept, int depth, boolean expanded, boolean hasChildren) {
            this.dept = dept;
            this.depth = depth;
            this.expanded = expanded;
            this.hasChildren = hasChildren;
        }
    }

    private final List<Node> flatNodes = new ArrayList<>();
    private final List<DeptVO> rootDepts = new ArrayList<>();
    private OnDeptActionListener actionListener;
    private final SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault());

    public void setOnDeptActionListener(OnDeptActionListener listener) {
        this.actionListener = listener;
    }

    public void setData(List<DeptVO> depts) {
        rootDepts.clear();
        if (depts != null) {
            rootDepts.addAll(depts);
        }
        rebuildFlatNodes();
        notifyDataSetChanged();
    }

    private void rebuildFlatNodes() {
        flatNodes.clear();
        for (DeptVO dept : rootDepts) {
            flatten(dept, 0, true);
        }
    }

    private void flatten(DeptVO dept, int depth, boolean parentExpanded) {
        if (!parentExpanded) {
            return;
        }
        boolean hasChildren = dept.getChildren() != null && !dept.getChildren().isEmpty();
        Node node = new Node(dept, depth, true, hasChildren);
        flatNodes.add(node);
        if (hasChildren && node.expanded) {
            for (DeptVO child : dept.getChildren()) {
                flatten(child, depth + 1, true);
            }
        }
    }

    private void rebuildAndNotify() {
        // 保留当前展开状态
        java.util.Map<Integer, Boolean> expandStates = new java.util.HashMap<>();
        for (Node node : flatNodes) {
            if (node.dept.getId() != null) {
                expandStates.put(node.dept.getId(), node.expanded);
            }
        }
        flatNodes.clear();
        for (DeptVO dept : rootDepts) {
            flattenWithState(dept, 0, true, expandStates);
        }
        notifyDataSetChanged();
    }

    private void flattenWithState(DeptVO dept, int depth, boolean parentExpanded, java.util.Map<Integer, Boolean> expandStates) {
        if (!parentExpanded) {
            return;
        }
        boolean hasChildren = dept.getChildren() != null && !dept.getChildren().isEmpty();
        Integer deptId = dept.getId();
        boolean expanded = deptId == null || expandStates.getOrDefault(deptId, true);
        Node node = new Node(dept, depth, expanded, hasChildren);
        flatNodes.add(node);
        if (hasChildren && expanded) {
            for (DeptVO child : dept.getChildren()) {
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
    public DeptViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_dept, parent, false);
        return new DeptViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull DeptViewHolder holder, int position) {
        Node node = flatNodes.get(position);
        holder.bind(node);
    }

    class DeptViewHolder extends RecyclerView.ViewHolder {
        private TextView tvName;
        private TextView tvStatus;
        private TextView tvSort;
        private TextView tvCreateTime;
        private TextView tvEdit;
        private TextView tvDelete;
        private TextView tvAddChild;
        private ImageView ivExpand;
        private View indentView;

        DeptViewHolder(@NonNull View itemView) {
            super(itemView);
            tvName = itemView.findViewById(R.id.tv_name);
            tvStatus = itemView.findViewById(R.id.tv_status);
            tvSort = itemView.findViewById(R.id.tv_sort);
            tvCreateTime = itemView.findViewById(R.id.tv_create_time);
            tvEdit = itemView.findViewById(R.id.tv_edit);
            tvDelete = itemView.findViewById(R.id.tv_delete);
            tvAddChild = itemView.findViewById(R.id.tv_add_child);
            ivExpand = itemView.findViewById(R.id.iv_expand);
            indentView = itemView.findViewById(R.id.indent);
        }

        void bind(Node node) {
            DeptVO dept = node.dept;
            tvName.setText(dept.getName() == null ? "" : dept.getName());
            Integer status = dept.getStatus();
            if (status != null && status == 1) {
                tvStatus.setText("启用");
                tvStatus.setTextColor(0xFF4CAF50);
            } else {
                tvStatus.setText("禁用");
                tvStatus.setTextColor(0xFF9E9E9E);
            }
            tvSort.setText(dept.getSort() != null ? String.valueOf(dept.getSort()) : "0");
            Date createTime = dept.getCreateTime();
            tvCreateTime.setText(createTime != null ? dateFormat.format(createTime) : "-");

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

            tvEdit.setOnClickListener(v -> {
                if (actionListener != null) actionListener.onEdit(dept);
            });
            tvDelete.setOnClickListener(v -> {
                if (actionListener != null) actionListener.onDelete(dept);
            });
            tvAddChild.setOnClickListener(v -> {
                if (actionListener != null) actionListener.onAddChild(dept);
            });
        }
    }
}
