package com.pei.dehaze.ui.dataset;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.CheckBox;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.pei.dehaze.R;
import com.pei.dehaze.sdk.model.dataset.Dataset;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * 数据集树形列表适配器（支持懒加载子节点、搜索扁平模式、选择模式）
 */
public class DatasetTreeAdapter extends RecyclerView.Adapter<DatasetTreeAdapter.DatasetViewHolder> {

    private static final int INDENT_PER_LEVEL = 48;

    /**
     * 扁平节点
     */
    private static class FlatNode {
        final Dataset dataset;
        final int depth;
        final boolean hasChildren;
        final boolean expanded;
        final boolean loaded;

        FlatNode(Dataset dataset, int depth, boolean hasChildren, boolean expanded, boolean loaded) {
            this.dataset = dataset;
            this.depth = depth;
            this.hasChildren = hasChildren;
            this.expanded = expanded;
            this.loaded = loaded;
        }
    }

    public interface OnDatasetActionListener {
        void onView(Dataset dataset);
        void onEdit(Dataset dataset);
        void onDelete(Dataset dataset);
        void onAddChild(Dataset parent);
        void onLazyLoadChildren(Dataset parent);
    }

    public interface OnSelectionChangedListener {
        void onSelectionChanged(Set<Long> selectedIds);
    }

    private final List<FlatNode> flatList = new ArrayList<>();
    private final List<Dataset> roots = new ArrayList<>();
    private final Map<Long, List<Dataset>> loadedChildren = new HashMap<>();
    private final Set<Long> expandedIds = new HashSet<>();
    private final Set<Long> selectedIds = new HashSet<>();

    private boolean treeMode = true;
    private boolean selectionMode = false;
    private boolean loadingChild = false;

    private OnDatasetActionListener actionListener;
    private OnSelectionChangedListener selectionListener;

    public void setActionListener(OnDatasetActionListener listener) {
        this.actionListener = listener;
    }

    public void setSelectionListener(OnSelectionChangedListener listener) {
        this.selectionListener = listener;
    }

    public void setTreeMode(boolean treeMode) {
        this.treeMode = treeMode;
        if (!treeMode) {
            expandedIds.clear();
            loadedChildren.clear();
        }
        rebuildFlatList();
        notifyDataSetChanged();
    }

    public void setRoots(List<Dataset> datasets) {
        roots.clear();
        loadedChildren.clear();
        expandedIds.clear();
        if (datasets != null) {
            roots.addAll(datasets);
        }
        treeMode = true;
        rebuildFlatList();
        notifyDataSetChanged();
    }

    public void setSearchResults(List<Dataset> datasets) {
        roots.clear();
        loadedChildren.clear();
        expandedIds.clear();
        if (datasets != null) {
            roots.addAll(datasets);
        }
        treeMode = false;
        rebuildFlatList();
        notifyDataSetChanged();
    }

    /**
     * 设置已加载的子节点（懒加载完成后调用）
     */
    public void setChildren(long parentId, List<Dataset> children) {
        loadedChildren.put(parentId, children != null ? children : new ArrayList<>());
        loadingChild = false;
        expandedIds.add(parentId);
        rebuildFlatList();
        notifyDataSetChanged();
    }

    public void setSelectionMode(boolean selectionMode) {
        this.selectionMode = selectionMode;
        if (!selectionMode) {
            selectedIds.clear();
            notifySelectionChanged();
        }
        notifyDataSetChanged();
    }

    public boolean isSelectionMode() {
        return selectionMode;
    }

    public void selectAll() {
        for (FlatNode node : flatList) {
            if (node.dataset.getId() != null) {
                selectedIds.add(node.dataset.getId());
            }
        }
        notifyDataSetChanged();
        notifySelectionChanged();
    }

    public void clearSelection() {
        selectedIds.clear();
        notifyDataSetChanged();
        notifySelectionChanged();
    }

    public Set<Long> getSelectedIds() {
        return new HashSet<>(selectedIds);
    }

    private void notifySelectionChanged() {
        if (selectionListener != null) {
            selectionListener.onSelectionChanged(new HashSet<>(selectedIds));
        }
    }

    private void rebuildFlatList() {
        flatList.clear();
        if (treeMode) {
            flatten(roots, 0);
        } else {
            for (Dataset dataset : roots) {
                flatList.add(new FlatNode(dataset, 0, false, false, true));
            }
        }
    }

    private void flatten(List<Dataset> nodes, int depth) {
        if (nodes == null) return;
        for (Dataset node : nodes) {
            Long id = node.getId();
            boolean hasChildren = Boolean.TRUE.equals(node.getHasChildren())
                    || (loadedChildren.containsKey(id) && !loadedChildren.get(id).isEmpty());
            boolean expanded = expandedIds.contains(id);
            boolean loaded = loadedChildren.containsKey(id);
            flatList.add(new FlatNode(node, depth, hasChildren, expanded, loaded));
            if (expanded && loaded) {
                flatten(loadedChildren.get(id), depth + 1);
            }
        }
    }

    private void toggleExpand(Dataset dataset) {
        Long id = dataset.getId();
        if (id == null) return;
        if (expandedIds.contains(id)) {
            expandedIds.remove(id);
            rebuildFlatList();
            notifyDataSetChanged();
        } else {
            if (loadedChildren.containsKey(id)) {
                expandedIds.add(id);
                rebuildFlatList();
                notifyDataSetChanged();
            } else if (!loadingChild) {
                loadingChild = true;
                if (actionListener != null) {
                    actionListener.onLazyLoadChildren(dataset);
                }
            }
        }
    }

    @NonNull
    @Override
    public DatasetViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_dataset, parent, false);
        return new DatasetViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull DatasetViewHolder holder, int position) {
        holder.bind(flatList.get(position));
    }

    @Override
    public int getItemCount() {
        return flatList.size();
    }

    class DatasetViewHolder extends RecyclerView.ViewHolder {
        private final CheckBox cbSelect;
        private final ImageView ivExpand;
        private final TextView tvName;
        private final TextView tvType;
        private final TextView tvStatus;
        private final TextView tvItemCount;
        private final TextView tvView;
        private final TextView tvEdit;
        private final TextView tvDelete;
        private final TextView tvAddChild;
        private final LinearLayout layoutActions;

        DatasetViewHolder(@NonNull View itemView) {
            super(itemView);
            cbSelect = itemView.findViewById(R.id.cb_select);
            ivExpand = itemView.findViewById(R.id.iv_expand);
            tvName = itemView.findViewById(R.id.tv_name);
            tvType = itemView.findViewById(R.id.tv_type);
            tvStatus = itemView.findViewById(R.id.tv_status);
            tvItemCount = itemView.findViewById(R.id.tv_item_count);
            tvView = itemView.findViewById(R.id.tv_view);
            tvEdit = itemView.findViewById(R.id.tv_edit);
            tvDelete = itemView.findViewById(R.id.tv_delete);
            tvAddChild = itemView.findViewById(R.id.tv_add_child);
            layoutActions = itemView.findViewById(R.id.layout_actions);
        }

        void bind(FlatNode node) {
            Dataset dataset = node.dataset;

            int padding = node.depth * INDENT_PER_LEVEL;
            itemView.setPadding(padding + 16, itemView.getPaddingTop(),
                    itemView.getPaddingRight(), itemView.getPaddingBottom());

            tvName.setText(safe(dataset.getName()));
            tvType.setText(safe(dataset.getType()));
            Integer status = dataset.getStatus();
            if (status != null && status == 1) {
                tvStatus.setText("启用");
                tvStatus.setTextColor(0xFF4CAF50);
            } else {
                tvStatus.setText("禁用");
                tvStatus.setTextColor(0xFF9E9E9E);
            }
            if (dataset.getStatistics() != null && dataset.getStatistics().getItemCount() != null) {
                tvItemCount.setText(dataset.getStatistics().getItemCount() + " 项");
            } else if (dataset.getTotal() != null) {
                tvItemCount.setText(dataset.getTotal() + " 项");
            } else {
                tvItemCount.setText("-");
            }

            if (treeMode) {
                if (node.hasChildren) {
                    ivExpand.setVisibility(View.VISIBLE);
                    if (node.expanded) {
                        ivExpand.setImageResource(R.drawable.ic_arrow_down);
                    } else {
                        ivExpand.setImageResource(R.drawable.ic_arrow_right);
                    }
                    ivExpand.setOnClickListener(v -> toggleExpand(dataset));
                } else {
                    ivExpand.setVisibility(View.INVISIBLE);
                    ivExpand.setOnClickListener(null);
                }
            } else {
                ivExpand.setVisibility(View.INVISIBLE);
                ivExpand.setOnClickListener(null);
            }

            if (selectionMode) {
                cbSelect.setVisibility(View.VISIBLE);
                layoutActions.setVisibility(View.GONE);
                Long id = dataset.getId();
                cbSelect.setOnCheckedChangeListener(null);
                cbSelect.setChecked(id != null && selectedIds.contains(id));
                cbSelect.setOnCheckedChangeListener((button, checked) -> {
                    if (id == null) {
                        cbSelect.setChecked(false);
                        return;
                    }
                    if (checked) {
                        selectedIds.add(id);
                    } else {
                        selectedIds.remove(id);
                    }
                    notifySelectionChanged();
                });
                itemView.setOnClickListener(v -> {
                    if (id == null) return;
                    boolean checked = !selectedIds.contains(id);
                    cbSelect.setChecked(checked);
                });
            } else {
                cbSelect.setVisibility(View.GONE);
                cbSelect.setOnCheckedChangeListener(null);
                layoutActions.setVisibility(View.VISIBLE);
                itemView.setOnClickListener(null);
                tvView.setOnClickListener(v -> {
                    if (actionListener != null) actionListener.onView(dataset);
                });
                tvEdit.setOnClickListener(v -> {
                    if (actionListener != null) actionListener.onEdit(dataset);
                });
                tvDelete.setOnClickListener(v -> {
                    if (actionListener != null) actionListener.onDelete(dataset);
                });
                tvAddChild.setOnClickListener(v -> {
                    if (actionListener != null) actionListener.onAddChild(dataset);
                });
            }
        }

        private String safe(String s) {
            return s == null ? "" : s;
        }
    }
}
