package com.pei.dehaze.ui.system;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.pei.dehaze.R;
import com.pei.dehaze.databinding.ActivityManageListBinding;
import com.pei.dehaze.sdk.model.member.MemberPageVO;
import com.pei.dehaze.ui.system.viewmodel.MemberManageViewModel;
import com.pei.dehaze.utils.StringUtils;
import com.pei.dehaze.utils.ToastUtils;

import java.util.List;

/**
 * 会员管理（sys:member:*）— 完整 CRUD：列表查看、等级调整、状态切换、禁用（删除）
 */
public class MemberManageActivity extends AppCompatActivity {

    private MemberManageViewModel viewModel;
    private ActivityManageListBinding binding;
    private MemberManageAdapter adapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityManageListBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());
        initViews();
        initViewModel();
        setupObservers();
        loadData();
    }

    private void initViews() {
        setSupportActionBar(binding.toolbar);
        binding.toolbar.setTitle("会员管理");
        binding.toolbar.setNavigationOnClickListener(v -> finish());

        ArrayAdapter<String> statusAdapter = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_item,
                new String[]{"全部", "普通会员", "VIP", "SVIP"});
        statusAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        binding.spinnerStatus.setAdapter(statusAdapter);

        adapter = new MemberManageAdapter();
        binding.recyclerView.setLayoutManager(new LinearLayoutManager(this));
        binding.recyclerView.setAdapter(adapter);

        binding.swipeRefresh.setOnRefreshListener(this::loadData);

        binding.btnSearch.setOnClickListener(v -> {
            viewModel.setKeywords(binding.etKeywords.getText().toString().trim());
            loadData();
        });

        binding.btnReset.setOnClickListener(v -> {
            binding.etKeywords.setText("");
            binding.spinnerStatus.setSelection(0);
            viewModel.resetQuery();
            loadData();
        });

        binding.btnPrev.setOnClickListener(v -> viewModel.prevPage());
        binding.btnNext.setOnClickListener(v -> viewModel.nextPage());
    }

    private void initViewModel() {
        viewModel = new ViewModelProvider(this).get(MemberManageViewModel.class);
    }

    private void setupObservers() {
        viewModel.getItemList().observe(this, items -> {
            adapter.notifyDataSetChanged();
            binding.tvEmpty.setVisibility(items == null || items.isEmpty() ? View.VISIBLE : View.GONE);
            updatePageInfo();
        });

        viewModel.getTotal().observe(this, total -> updatePageInfo());

        viewModel.getLoading().observe(this, isLoading ->
                binding.swipeRefresh.setRefreshing(isLoading != null && isLoading));

        viewModel.getError().observe(this, errorMessage -> {
            if (errorMessage != null && !errorMessage.isEmpty()) {
                ToastUtils.showShort(this, errorMessage);
                viewModel.clearError();
            }
        });

        viewModel.getOperationResult().observe(this, result -> {
            if (result != null && !result.isEmpty()) {
                ToastUtils.showShort(this, result);
                viewModel.clearOperationResult();
            }
        });
    }

    private void loadData() {
        viewModel.loadData();
    }

    private void updatePageInfo() {
        long total = viewModel.getTotal().getValue() != null ? viewModel.getTotal().getValue() : 0L;
        int pageNum = viewModel.getPageNum();
        int pageSize = viewModel.getPageSize();
        int totalPages = Math.max(1, (int) Math.ceil(total * 1.0 / pageSize));
        binding.tvPageInfo.setText("第 " + pageNum + " 页 / 共 " + totalPages + " 页 (共 " + total + " 条)");
    }

    // ---- 等级调整对话框 ----
    private void showAdjustLevelDialog(MemberPageVO item) {
        if (item.getUserId() == null) return;
        final String[] levelCodes = {"bronze", "silver", "gold", "platinum", "diamond"};
        final String[] levelNames = {"青铜会员", "白银会员", "黄金会员", "铂金会员", "钻石会员"};

        String displayName = StringUtils.safe(item.getNickname(),
                StringUtils.safe(item.getUsername(), "--"));
        new AlertDialog.Builder(this)
                .setTitle("调整会员等级 — " + displayName)
                .setItems(levelNames, (dialog, which) -> {
                    String code = levelCodes[which];
                    String name = levelNames[which];
                    new AlertDialog.Builder(MemberManageActivity.this)
                            .setTitle("确认调整")
                            .setMessage("确认将 " + displayName
                                    + " 的等级调整为「" + name + "」吗？")
                            .setPositiveButton("确定", (d, w) ->
                                    viewModel.adjustLevel(item.getUserId(), code, "管理员手动调整"))
                            .setNegativeButton("取消", null)
                            .show();
                })
                .setNegativeButton("取消", null)
                .show();
    }

    // ---- 状态切换确认 ----
    private void showToggleStatusDialog(MemberPageVO item) {
        if (item.getUserId() == null) return;
        int currentStatus = item.getStatus() != null ? item.getStatus() : 1;
        int newStatus = currentStatus == 1 ? 0 : 1;
        String action = newStatus == 1 ? "启用" : "禁用";
        String displayName = StringUtils.safe(item.getNickname(),
                StringUtils.safe(item.getUsername(), "--"));

        new AlertDialog.Builder(this)
                .setTitle("确认" + action)
                .setMessage("确认" + action + "会员「" + displayName + "」吗？")
                .setPositiveButton("确定", (dialog, which) ->
                        viewModel.updateStatus(item.getUserId(), newStatus))
                .setNegativeButton("取消", null)
                .show();
    }

    // ---- 禁用（删除）确认 ----
    private void showDisableConfirmDialog(MemberPageVO item) {
        if (item.getUserId() == null) return;
        String displayName = StringUtils.safe(item.getNickname(),
                StringUtils.safe(item.getUsername(), "--"));
        new AlertDialog.Builder(this)
                .setTitle("禁用确认")
                .setMessage("确认禁用会员「" + displayName + "」吗？禁用后该用户将无法享受会员权益。")
                .setPositiveButton("确定", (dialog, which) ->
                        viewModel.deleteMember(item.getUserId()))
                .setNegativeButton("取消", null)
                .show();
    }

    // ---- Adapter ----
    private class MemberManageAdapter extends RecyclerView.Adapter<MemberManageAdapter.ViewHolder> {

        @NonNull
        @Override
        public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
            View view = LayoutInflater.from(parent.getContext())
                    .inflate(R.layout.item_member_manage, parent, false);
            return new ViewHolder(view);
        }

        @Override
        public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
            List<?> list = viewModel.getItemList().getValue();
            if (list == null || position >= list.size()) return;
            Object obj = list.get(position);
            if (!(obj instanceof MemberPageVO)) return;
            MemberPageVO item = (MemberPageVO) obj;

            String name = StringUtils.safe(item.getNickname(),
                    StringUtils.safe(item.getUsername(), "--"));
            holder.tvUsername.setText(name);

            String level = item.getLevelName() != null ? item.getLevelName() : "--";
            holder.tvLevel.setText("等级: " + level);

            String growth = "成长值: " + (item.getGrowthValue() != null ? item.getGrowthValue() : 0);
            holder.tvGrowth.setText(growth);

            int status = item.getStatus() != null ? item.getStatus() : 0;
            holder.tvStatus.setText(status == 1 ? "启用" : "禁用");
            holder.tvStatus.setTextColor(status == 1 ? 0xFF4CAF50 : 0xFFF44336);

            String expire = item.getExpireTime() != null ? "到期: " + item.getExpireTime() : "永久有效";
            holder.tvExpire.setText(expire);

            String monthly = "月已用: " + (item.getMonthlyUsed() != null ? item.getMonthlyUsed() : 0);
            holder.tvMonthlyUsed.setText(monthly);

            holder.tvAdjustLevel.setOnClickListener(v -> showAdjustLevelDialog(item));
            holder.tvToggleStatus.setOnClickListener(v -> showToggleStatusDialog(item));
            holder.tvDelete.setOnClickListener(v -> showDisableConfirmDialog(item));
        }

        @Override
        public int getItemCount() {
            List<?> list = viewModel.getItemList().getValue();
            return list != null ? list.size() : 0;
        }

        class ViewHolder extends RecyclerView.ViewHolder {
            TextView tvUsername, tvLevel, tvGrowth, tvStatus, tvExpire, tvMonthlyUsed;
            TextView tvAdjustLevel, tvToggleStatus, tvDelete;

            ViewHolder(View itemView) {
                super(itemView);
                tvUsername = itemView.findViewById(R.id.tv_username);
                tvLevel = itemView.findViewById(R.id.tv_level);
                tvGrowth = itemView.findViewById(R.id.tv_growth);
                tvStatus = itemView.findViewById(R.id.tv_status);
                tvExpire = itemView.findViewById(R.id.tv_expire);
                tvMonthlyUsed = itemView.findViewById(R.id.tv_monthly_used);
                tvAdjustLevel = itemView.findViewById(R.id.tv_adjust_level);
                tvToggleStatus = itemView.findViewById(R.id.tv_toggle_status);
                tvDelete = itemView.findViewById(R.id.tv_delete);
            }
        }
    }
}
