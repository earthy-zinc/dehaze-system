package com.pei.dehaze.ui.system.adapter;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.pei.dehaze.R;
import com.pei.dehaze.sdk.model.menu.MenuVO;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class MenuAdapter extends RecyclerView.Adapter<MenuAdapter.MenuViewHolder> {

    private static final int INDENT_PER_LEVEL = 40;

    private final List<FlatMenuItem> flatList = new ArrayList<>();
    private List<MenuVO> treeData = new ArrayList<>();
    private final Set<Integer> expandedIds = new HashSet<>();
    private OnMenuActionListener listener;

    public static class FlatMenuItem {
        public final MenuVO menu;
        public final int depth;
        public final boolean hasChildren;
        public final boolean expanded;

        FlatMenuItem(MenuVO menu, int depth, boolean hasChildren, boolean expanded) {
            this.menu = menu;
            this.depth = depth;
            this.hasChildren = hasChildren;
            this.expanded = expanded;
        }
    }

    public interface OnMenuActionListener {
        void onEdit(MenuVO menu);
        void onDelete(MenuVO menu);
    }

    public void setListener(OnMenuActionListener listener) {
        this.listener = listener;
    }

    public void setMenuTree(List<MenuVO> tree) {
        this.treeData = tree != null ? tree : new ArrayList<>();
        if (expandedIds.isEmpty() && !this.treeData.isEmpty()) {
            for (MenuVO node : this.treeData) {
                expandedIds.add(node.getId());
            }
        }
        rebuildFlatList();
        notifyDataSetChanged();
    }

    private void rebuildFlatList() {
        flatList.clear();
        flatten(treeData, 0);
    }

    private void flatten(List<MenuVO> nodes, int depth) {
        if (nodes == null) return;
        for (MenuVO node : nodes) {
            boolean hasChildren = node.getChildren() != null && !node.getChildren().isEmpty();
            boolean expanded = expandedIds.contains(node.getId());
            flatList.add(new FlatMenuItem(node, depth, hasChildren, expanded));
            if (expanded && hasChildren) {
                flatten(node.getChildren(), depth + 1);
            }
        }
    }

    @NonNull
    @Override
    public MenuViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_menu, parent, false);
        return new MenuViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull MenuViewHolder holder, int position) {
        holder.bind(flatList.get(position));
    }

    @Override
    public int getItemCount() {
        return flatList.size();
    }

    class MenuViewHolder extends RecyclerView.ViewHolder {
        private final ImageView ivExpand;
        private final TextView tvName;
        private final TextView tvType;
        private final TextView tvPath;
        private final TextView tvPerm;
        private final TextView tvVisible;
        private final TextView tvSort;
        private final ImageButton btnEdit;
        private final ImageButton btnDelete;

        MenuViewHolder(@NonNull View itemView) {
            super(itemView);
            ivExpand = itemView.findViewById(R.id.iv_expand);
            tvName = itemView.findViewById(R.id.tv_name);
            tvType = itemView.findViewById(R.id.tv_type);
            tvPath = itemView.findViewById(R.id.tv_path);
            tvPerm = itemView.findViewById(R.id.tv_perm);
            tvVisible = itemView.findViewById(R.id.tv_visible);
            tvSort = itemView.findViewById(R.id.tv_sort);
            btnEdit = itemView.findViewById(R.id.btn_edit);
            btnDelete = itemView.findViewById(R.id.btn_delete);
        }

        void bind(FlatMenuItem item) {
            int padding = item.depth * INDENT_PER_LEVEL;
            itemView.setPadding(padding + 16, itemView.getPaddingTop(),
                    itemView.getPaddingRight(), itemView.getPaddingBottom());

            MenuVO menu = item.menu;
            tvName.setText(menu.getName());
            tvType.setText(typeLabel(menu.getType()));
            tvPath.setText(menu.getPath() != null ? menu.getPath() : "");
            tvPerm.setText(menu.getPerm() != null ? menu.getPerm() : "");
            tvVisible.setText(menu.getVisible() != null && menu.getVisible() == 1 ? "显示" : "隐藏");
            tvSort.setText(menu.getSort() != null ? String.valueOf(menu.getSort()) : "");

            if (item.hasChildren) {
                ivExpand.setVisibility(View.VISIBLE);
                ivExpand.setImageResource(item.expanded
                        ? R.drawable.ic_arrow_down
                        : R.drawable.ic_arrow_right);
            } else {
                ivExpand.setVisibility(View.INVISIBLE);
            }

            ivExpand.setOnClickListener(v -> {
                if (item.expanded) {
                    expandedIds.remove(item.menu.getId());
                } else {
                    expandedIds.add(item.menu.getId());
                }
                rebuildFlatList();
                notifyDataSetChanged();
            });

            btnEdit.setOnClickListener(v -> {
                if (listener != null) {
                    listener.onEdit(menu);
                }
            });

            btnDelete.setOnClickListener(v -> {
                if (listener != null) {
                    listener.onDelete(menu);
                }
            });
        }

        private String typeLabel(String type) {
            if (type == null) return "";
            switch (type) {
                case "CATALOG": return "目录";
                case "MENU": return "菜单";
                case "BUTTON": return "按钮";
                case "EXTLINK": return "外链";
                default: return type;
            }
        }
    }
}
