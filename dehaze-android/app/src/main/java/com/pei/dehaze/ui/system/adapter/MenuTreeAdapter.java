package com.pei.dehaze.ui.system.adapter;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.CheckBox;
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

/**
 * 菜单权限树适配器，用于角色权限分配。
 * 支持展开/收起和父子联动勾选。
 */
public class MenuTreeAdapter extends RecyclerView.Adapter<MenuTreeAdapter.MenuViewHolder> {

    private static class Node {
        MenuVO menu;
        int depth;
        boolean expanded;
        boolean hasChildren;
        Node(MenuVO menu, int depth, boolean expanded, boolean hasChildren) {
            this.menu = menu;
            this.depth = depth;
            this.expanded = expanded;
            this.hasChildren = hasChildren;
        }
    }

    private final List<Node> flatNodes = new ArrayList<>();
    private final List<MenuVO> rootMenus = new ArrayList<>();
    private final Set<Integer> checkedIds = new HashSet<>();
    private final Set<Integer> indeterminateIds = new HashSet<>();

    public void setData(List<MenuVO> menus) {
        rootMenus.clear();
        if (menus != null) {
            rootMenus.addAll(menus);
        }
        rebuildFlatNodes();
        notifyDataSetChanged();
    }

    public void setCheckedIds(List<Integer> ids) {
        checkedIds.clear();
        if (ids != null) {
            checkedIds.addAll(ids);
        }
        rebuildFlatNodes();
        notifyDataSetChanged();
    }

    public List<Integer> getCheckedIds() {
        return new ArrayList<>(checkedIds);
    }

    private void rebuildFlatNodes() {
        java.util.Map<Integer, Boolean> expandStates = new java.util.HashMap<>();
        for (Node node : flatNodes) {
            if (node.menu.getId() != null) {
                expandStates.put(node.menu.getId(), node.expanded);
            }
        }
        flatNodes.clear();
        for (MenuVO menu : rootMenus) {
            flatten(menu, 0, true, expandStates);
        }
    }

    private void flatten(MenuVO menu, int depth, boolean parentExpanded, java.util.Map<Integer, Boolean> expandStates) {
        if (!parentExpanded) {
            return;
        }
        boolean hasChildren = menu.getChildren() != null && !menu.getChildren().isEmpty();
        Integer menuId = menu.getId();
        boolean expanded = menuId == null || expandStates.getOrDefault(menuId, true);
        Node node = new Node(menu, depth, expanded, hasChildren);
        flatNodes.add(node);
        if (hasChildren && expanded) {
            for (MenuVO child : menu.getChildren()) {
                flatten(child, depth + 1, true, expandStates);
            }
        }
    }

    @Override
    public int getItemCount() {
        return flatNodes.size();
    }

    @NonNull
    @Override
    public MenuViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_permission_tree, parent, false);
        return new MenuViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull MenuViewHolder holder, int position) {
        Node node = flatNodes.get(position);
        holder.bind(node);
    }

    private void toggleNodeExpanded(Node node) {
        node.expanded = !node.expanded;
        rebuildFlatNodes();
        notifyDataSetChanged();
    }

    private void setSubtreeChecked(MenuVO menu, boolean checked) {
        Integer id = menu.getId();
        if (id != null) {
            if (checked) {
                checkedIds.add(id);
            } else {
                checkedIds.remove(id);
            }
        }
        if (menu.getChildren() != null) {
            for (MenuVO child : menu.getChildren()) {
                setSubtreeChecked(child, checked);
            }
        }
    }

    private void recomputeIndeterminateStates() {
        indeterminateIds.clear();
        for (MenuVO root : rootMenus) {
            recomputeIndeterminate(root);
        }
    }

    private boolean recomputeIndeterminate(MenuVO menu) {
        boolean allChecked = true;
        boolean anyChecked = false;
        Integer id = menu.getId();
        if (id != null) {
            if (checkedIds.contains(id)) {
                anyChecked = true;
            } else {
                allChecked = false;
            }
        }
        if (menu.getChildren() != null && !menu.getChildren().isEmpty()) {
            for (MenuVO child : menu.getChildren()) {
                boolean childAllChecked = recomputeIndeterminate(child);
                if (!childAllChecked) {
                    allChecked = false;
                }
                if (checkedIds.contains(child.getId()) || indeterminateIds.contains(child.getId())) {
                    anyChecked = true;
                }
            }
        }
        // 子项部分选中 -> 父项设为半选并加入 checked 以向上传递选中状态
        if (id != null) {
            if (!allChecked && anyChecked) {
                indeterminateIds.add(id);
                checkedIds.add(id);
            } else {
                indeterminateIds.remove(id);
                if (!allChecked) {
                    checkedIds.remove(id);
                }
            }
        }
        return allChecked;
    }

    class MenuViewHolder extends RecyclerView.ViewHolder {
        private CheckBox cbMenu;
        private TextView tvName;
        private TextView tvPerm;
        private ImageView ivExpand;
        private View indentView;

        MenuViewHolder(@NonNull View itemView) {
            super(itemView);
            cbMenu = itemView.findViewById(R.id.cb_menu);
            tvName = itemView.findViewById(R.id.tv_name);
            tvPerm = itemView.findViewById(R.id.tv_perm);
            ivExpand = itemView.findViewById(R.id.iv_expand);
            indentView = itemView.findViewById(R.id.indent);
        }

        void bind(Node node) {
            MenuVO menu = node.menu;
            tvName.setText(menu.getName() == null ? "" : menu.getName());
            tvPerm.setText(menu.getPerm() == null ? "" : menu.getPerm());
            tvPerm.setVisibility(menu.getPerm() == null || menu.getPerm().isEmpty() ? View.GONE : View.VISIBLE);

            int padding = node.depth * 32;
            indentView.getLayoutParams().width = padding;
            indentView.requestLayout();

            if (node.hasChildren) {
                ivExpand.setVisibility(View.VISIBLE);
                ivExpand.setImageResource(node.expanded ? R.drawable.ic_arrow_down : R.drawable.ic_arrow_right);
                ivExpand.setOnClickListener(v -> toggleNodeExpanded(node));
            } else {
                ivExpand.setVisibility(View.INVISIBLE);
                ivExpand.setOnClickListener(null);
            }

            // 避免在 setChecked 时触发回调
            cbMenu.setOnCheckedChangeListener(null);
            Integer menuId = menu.getId();
            boolean checked = menuId != null && checkedIds.contains(menuId);
            cbMenu.setChecked(checked);

            cbMenu.setOnClickListener(v -> {
                boolean newChecked = cbMenu.isChecked();
                setSubtreeChecked(menu, newChecked);
                recomputeIndeterminateStates();
                rebuildFlatNodes();
                notifyDataSetChanged();
            });
        }
    }
}
