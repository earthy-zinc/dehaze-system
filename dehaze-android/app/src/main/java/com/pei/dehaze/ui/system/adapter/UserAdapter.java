package com.pei.dehaze.ui.system.adapter;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.CheckBox;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.DiffUtil;
import androidx.recyclerview.widget.ListAdapter;
import androidx.recyclerview.widget.RecyclerView;

import com.pei.dehaze.R;
import com.pei.dehaze.sdk.model.user.UserPageVO;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

public class UserAdapter extends ListAdapter<UserPageVO, UserAdapter.UserViewHolder> {

    public interface OnUserActionListener {
        void onEdit(UserPageVO user);
        void onDelete(UserPageVO user);
        void onResetPassword(UserPageVO user);
        void onToggleStatus(UserPageVO user);
    }

    public interface OnSelectionChangedListener {
        void onSelectionChanged(Set<Integer> selectedIds);
    }

    private OnUserActionListener actionListener;
    private OnSelectionChangedListener selectionListener;
    private boolean selectionMode = false;
    private final Set<Integer> selectedIds = new HashSet<>();
    private final SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault());

    public UserAdapter() {
        super(DIFF_CALLBACK);
    }

    private static final DiffUtil.ItemCallback<UserPageVO> DIFF_CALLBACK = new DiffUtil.ItemCallback<UserPageVO>() {
        @Override
        public boolean areItemsTheSame(@NonNull UserPageVO oldItem, @NonNull UserPageVO newItem) {
            return oldItem.getId() != null && oldItem.getId().equals(newItem.getId());
        }

        @Override
        public boolean areContentsTheSame(@NonNull UserPageVO oldItem, @NonNull UserPageVO newItem) {
            return UserAdapter.equals(oldItem.getUsername(), newItem.getUsername()) &&
                   UserAdapter.equals(oldItem.getNickname(), newItem.getNickname()) &&
                   UserAdapter.equals(oldItem.getMobile(), newItem.getMobile()) &&
                   UserAdapter.equals(oldItem.getStatus(), newItem.getStatus()) &&
                   UserAdapter.equals(oldItem.getDeptName(), newItem.getDeptName());
        }
    };

    private static boolean equals(Object a, Object b) {
        return a == null ? b == null : a.equals(b);
    }

    public void setOnUserActionListener(OnUserActionListener listener) {
        this.actionListener = listener;
    }

    public void setOnSelectionChangedListener(OnSelectionChangedListener listener) {
        this.selectionListener = listener;
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
        List<UserPageVO> list = getCurrentList();
        for (UserPageVO user : list) {
            if (user.getId() != null) {
                selectedIds.add(user.getId());
            }
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

    public List<Integer> getSelectedIdList() {
        return new ArrayList<>(selectedIds);
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

    @NonNull
    @Override
    public UserViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_user, parent, false);
        return new UserViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull UserViewHolder holder, int position) {
        holder.bind(getItem(position));
    }

    class UserViewHolder extends RecyclerView.ViewHolder {
        private CheckBox cbSelect;
        private TextView tvUsername;
        private TextView tvNickname;
        private TextView tvDept;
        private TextView tvMobile;
        private TextView tvStatus;
        private TextView tvCreateTime;
        private TextView tvEdit;
        private TextView tvDelete;
        private TextView tvResetPassword;
        private TextView tvToggleStatus;

        UserViewHolder(@NonNull View itemView) {
            super(itemView);
            cbSelect = itemView.findViewById(R.id.cb_select);
            tvUsername = itemView.findViewById(R.id.tv_username);
            tvNickname = itemView.findViewById(R.id.tv_nickname);
            tvDept = itemView.findViewById(R.id.tv_dept);
            tvMobile = itemView.findViewById(R.id.tv_mobile);
            tvStatus = itemView.findViewById(R.id.tv_status);
            tvCreateTime = itemView.findViewById(R.id.tv_create_time);
            tvEdit = itemView.findViewById(R.id.tv_edit);
            tvDelete = itemView.findViewById(R.id.tv_delete);
            tvResetPassword = itemView.findViewById(R.id.tv_reset_password);
            tvToggleStatus = itemView.findViewById(R.id.tv_toggle_status);
        }

        void bind(UserPageVO user) {
            tvUsername.setText(safe(user.getUsername()));
            tvNickname.setText(safe(user.getNickname()));
            tvDept.setText(safe(user.getDeptName()));
            tvMobile.setText(safe(user.getMobile()));
            Integer status = user.getStatus();
            if (status != null && status == 1) {
                tvStatus.setText("启用");
                tvStatus.setTextColor(0xFF4CAF50);
                tvToggleStatus.setText("禁用");
            } else {
                tvStatus.setText("禁用");
                tvStatus.setTextColor(0xFF9E9E9E);
                tvToggleStatus.setText("启用");
            }
            Date createTime = user.getCreateTime();
            tvCreateTime.setText(createTime != null ? dateFormat.format(createTime) : "-");

            if (selectionMode) {
                cbSelect.setVisibility(View.VISIBLE);
                cbSelect.setChecked(user.getId() != null && selectedIds.contains(user.getId()));
                cbSelect.setOnCheckedChangeListener((button, checked) -> {
                    if (user.getId() == null) {
                        cbSelect.setChecked(false);
                        return;
                    }
                    if (checked) {
                        selectedIds.add(user.getId());
                    } else {
                        selectedIds.remove(user.getId());
                    }
                    notifySelectionChanged();
                });
                hideActions();
            } else {
                cbSelect.setVisibility(View.GONE);
                cbSelect.setOnCheckedChangeListener(null);
                showActions();
            }

            tvEdit.setOnClickListener(v -> {
                if (actionListener != null) actionListener.onEdit(user);
            });
            tvDelete.setOnClickListener(v -> {
                if (actionListener != null) actionListener.onDelete(user);
            });
            tvResetPassword.setOnClickListener(v -> {
                if (actionListener != null) actionListener.onResetPassword(user);
            });
            tvToggleStatus.setOnClickListener(v -> {
                if (actionListener != null) actionListener.onToggleStatus(user);
            });

            itemView.setOnLongClickListener(v -> {
                if (!selectionMode && user.getId() != null) {
                    setSelectionMode(true);
                    selectedIds.add(user.getId());
                    notifyItemRangeChanged(0, getItemCount());
                    notifySelectionChanged();
                    return true;
                }
                return false;
            });
        }

        private void showActions() {
            tvEdit.setVisibility(View.VISIBLE);
            tvDelete.setVisibility(View.VISIBLE);
            tvResetPassword.setVisibility(View.VISIBLE);
            tvToggleStatus.setVisibility(View.VISIBLE);
        }

        private void hideActions() {
            tvEdit.setVisibility(View.GONE);
            tvDelete.setVisibility(View.GONE);
            tvResetPassword.setVisibility(View.GONE);
            tvToggleStatus.setVisibility(View.GONE);
        }

        private String safe(String s) {
            return s == null ? "" : s;
        }
    }
}
