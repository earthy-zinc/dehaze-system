package com.pei.dehaze.ui.system;

import android.os.Bundle;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.EditText;
import android.widget.RadioGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.pei.dehaze.R;
import com.pei.dehaze.databinding.ActivityManageListBinding;
import com.pei.dehaze.sdk.model.pkg.PackageForm;
import com.pei.dehaze.sdk.model.pkg.PackagePageVO;
import com.pei.dehaze.ui.system.viewmodel.PackageManageViewModel;
import com.pei.dehaze.utils.StringUtils;
import com.pei.dehaze.utils.ToastUtils;

import java.util.List;

/**
 * 套餐管理（sys:package:*）— 完整 CRUD：列表、新增/编辑、上下架、删除
 */
public class PackageManageActivity extends AppCompatActivity {

    private PackageManageViewModel viewModel;
    private ActivityManageListBinding binding;
    private PackageManageAdapter adapter;

    private static final String[] LEVEL_CODES = {"bronze", "silver", "gold", "platinum", "diamond"};
    private static final String[] LEVEL_NAMES = {"青铜", "白银", "黄金", "铂金", "钻石"};
    private static final String[] PERIODS = {"monthly", "quarterly", "yearly", "lifetime"};
    private static final String[] PERIOD_NAMES = {"月卡", "季卡", "年卡", "永久"};

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
        binding.toolbar.setTitle("套餐管理");
        binding.toolbar.setNavigationOnClickListener(v -> finish());

        ArrayAdapter<String> statusAdapter = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_item,
                new String[]{"全部", "上架", "下架"});
        statusAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        binding.spinnerStatus.setAdapter(statusAdapter);

        adapter = new PackageManageAdapter();
        binding.recyclerView.setLayoutManager(new LinearLayoutManager(this));
        binding.recyclerView.setAdapter(adapter);

        binding.swipeRefresh.setOnRefreshListener(this::loadData);

        // 显示新增按钮
        binding.btnAdd.setVisibility(View.VISIBLE);
        binding.btnAdd.setOnClickListener(v -> showPackageFormDialog(null));

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
        viewModel = new ViewModelProvider(this).get(PackageManageViewModel.class);
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

    // ---- 新增/编辑对话框 ----
    private void showPackageFormDialog(PackagePageVO existing) {
        boolean isEdit = existing != null;
        View formView = LayoutInflater.from(this).inflate(R.layout.dialog_package_form, null);

        EditText etName = formView.findViewById(R.id.et_name);
        EditText etOriginalPrice = formView.findViewById(R.id.et_original_price);
        EditText etSalePrice = formView.findViewById(R.id.et_sale_price);
        EditText etPeriodDays = formView.findViewById(R.id.et_period_days);
        RadioGroup rgStatus = formView.findViewById(R.id.rg_status);

        // 临时选中值
        int[] selectedLevelIdx = {0};
        int[] selectedPeriodIdx = {0};

        if (isEdit) {
            etName.setText(StringUtils.safe(existing.getName()));
            etOriginalPrice.setText(existing.getOriginalPrice() != null ? String.valueOf(existing.getOriginalPrice()) : "");
            etSalePrice.setText(existing.getSalePrice() != null ? String.valueOf(existing.getSalePrice()) : "");
            etPeriodDays.setText(existing.getPeriodDays() != null ? String.valueOf(existing.getPeriodDays()) : "");
            rgStatus.check(existing.getStatus() != null && existing.getStatus() == 1
                    ? R.id.rb_status_on : R.id.rb_status_off);

            // 匹配当前等级和周期
            for (int i = 0; i < LEVEL_CODES.length; i++) {
                if (LEVEL_CODES[i].equals(existing.getLevelCode())) {
                    selectedLevelIdx[0] = i;
                    break;
                }
            }
            for (int i = 0; i < PERIODS.length; i++) {
                if (PERIODS[i].equals(existing.getPeriod())) {
                    selectedPeriodIdx[0] = i;
                    break;
                }
            }
        } else {
            rgStatus.check(R.id.rb_status_on);
        }

        // 等级选择按钮
        TextView tvLevel = formView.findViewById(R.id.tv_level);
        tvLevel.setText("会员等级: " + LEVEL_NAMES[selectedLevelIdx[0]]);
        tvLevel.setOnClickListener(v -> {
            new AlertDialog.Builder(PackageManageActivity.this)
                    .setTitle("选择会员等级")
                    .setItems(LEVEL_NAMES, (d, w) -> {
                        selectedLevelIdx[0] = w;
                        tvLevel.setText("会员等级: " + LEVEL_NAMES[w]);
                    })
                    .show();
        });

        // 周期选择按钮
        TextView tvPeriod = formView.findViewById(R.id.tv_period);
        tvPeriod.setText("周期: " + PERIOD_NAMES[selectedPeriodIdx[0]]);
        tvPeriod.setOnClickListener(v -> {
            new AlertDialog.Builder(PackageManageActivity.this)
                    .setTitle("选择周期")
                    .setItems(PERIOD_NAMES, (d, w) -> {
                        selectedPeriodIdx[0] = w;
                        tvPeriod.setText("周期: " + PERIOD_NAMES[w]);
                    })
                    .show();
        });

        new AlertDialog.Builder(this)
                .setTitle(isEdit ? "编辑套餐" : "新增套餐")
                .setView(formView)
                .setPositiveButton("确定", (dialog, which) -> {
                    String name = etName.getText().toString().trim();
                    String origPriceStr = etOriginalPrice.getText().toString().trim();
                    String salePriceStr = etSalePrice.getText().toString().trim();
                    String periodDaysStr = etPeriodDays.getText().toString().trim();
                    int status = rgStatus.getCheckedRadioButtonId() == R.id.rb_status_on ? 1 : 0;

                    if (TextUtils.isEmpty(name)) {
                        ToastUtils.showShort(this, "请输入套餐名称");
                        return;
                    }

                    PackageForm form = new PackageForm();
                    form.setName(name);
                    form.setLevelCode(LEVEL_CODES[selectedLevelIdx[0]]);
                    form.setPeriod(PERIODS[selectedPeriodIdx[0]]);
                    form.setStatus(status);

                    if (!TextUtils.isEmpty(origPriceStr)) {
                        try { form.setOriginalPrice(Double.parseDouble(origPriceStr)); }
                        catch (NumberFormatException e) { /* ignore */ }
                    }
                    if (!TextUtils.isEmpty(salePriceStr)) {
                        try { form.setSalePrice(Double.parseDouble(salePriceStr)); }
                        catch (NumberFormatException e) { /* ignore */ }
                    }
                    if (!TextUtils.isEmpty(periodDaysStr)) {
                        try { form.setPeriodDays(Integer.parseInt(periodDaysStr)); }
                        catch (NumberFormatException e) { /* ignore */ }
                    }

                    if (isEdit) {
                        form.setId(existing.getId());
                        viewModel.updatePackage(existing.getId(), form);
                    } else {
                        viewModel.addPackage(form);
                    }
                })
                .setNegativeButton("取消", null)
                .show();
    }

    // ---- 上下架切换 ----
    private void showToggleStatusDialog(PackagePageVO item) {
        if (item.getId() == null) return;
        int currentStatus = item.getStatus() != null ? item.getStatus() : 1;
        int newStatus = currentStatus == 1 ? 0 : 1;
        String action = newStatus == 1 ? "上架" : "下架";

        new AlertDialog.Builder(this)
                .setTitle("确认" + action)
                .setMessage("确认" + action + "套餐「" + StringUtils.safe(item.getName(), "未命名") + "」吗？")
                .setPositiveButton("确定", (dialog, which) ->
                        viewModel.toggleStatus(item.getId(), newStatus))
                .setNegativeButton("取消", null)
                .show();
    }

    // ---- 删除确认 ----
    private void showDeleteConfirmDialog(PackagePageVO item) {
        if (item.getId() == null) return;
        new AlertDialog.Builder(this)
                .setTitle("删除确认")
                .setMessage("确认删除套餐「" + StringUtils.safe(item.getName(), "未命名") + "」吗？删除后不可恢复。")
                .setPositiveButton("确定", (dialog, which) ->
                        viewModel.deletePackage(item.getId()))
                .setNegativeButton("取消", null)
                .show();
    }

    // ---- Adapter ----
    private class PackageManageAdapter extends RecyclerView.Adapter<PackageManageAdapter.ViewHolder> {

        @NonNull
        @Override
        public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
            View view = LayoutInflater.from(parent.getContext())
                    .inflate(R.layout.item_package_manage, parent, false);
            return new ViewHolder(view);
        }

        @Override
        public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
            List<?> list = viewModel.getItemList().getValue();
            if (list == null || position >= list.size()) return;
            Object obj = list.get(position);
            if (!(obj instanceof PackagePageVO)) return;
            PackagePageVO item = (PackagePageVO) obj;

            holder.tvName.setText(StringUtils.safe(item.getName(), "未命名套餐"));

            int status = item.getStatus() != null ? item.getStatus() : 0;
            holder.tvStatus.setText(status == 1 ? "上架" : "下架");
            holder.tvStatus.setTextColor(status == 1 ? 0xFF4CAF50 : 0xFFF44336);

            if (item.getSalePrice() != null) {
                holder.tvPrice.setText("¥" + String.format("%.2f", item.getSalePrice()));
                holder.tvPrice.setVisibility(View.VISIBLE);
            } else {
                holder.tvPrice.setVisibility(View.GONE);
            }

            if (item.getOriginalPrice() != null && item.getOriginalPrice() > 0) {
                holder.tvOriginalPrice.setText("原价¥" + String.format("%.2f", item.getOriginalPrice()));
                holder.tvOriginalPrice.setVisibility(View.VISIBLE);
            } else {
                holder.tvOriginalPrice.setVisibility(View.GONE);
            }

            holder.tvPeriod.setText(StringUtils.safe(item.getPeriod(), "--"));

            holder.tvLevel.setText("等级: " + StringUtils.safe(item.getLevelName(), "--"));

            String sales = "销量: " + (item.getSalesCount() != null ? item.getSalesCount() : 0);
            holder.tvSales.setText(sales);

            holder.tvEdit.setOnClickListener(v -> showPackageFormDialog(item));
            holder.tvToggleStatus.setOnClickListener(v -> showToggleStatusDialog(item));
            holder.tvDelete.setOnClickListener(v -> showDeleteConfirmDialog(item));
        }

        @Override
        public int getItemCount() {
            List<?> list = viewModel.getItemList().getValue();
            return list != null ? list.size() : 0;
        }

        class ViewHolder extends RecyclerView.ViewHolder {
            TextView tvName, tvStatus, tvPrice, tvOriginalPrice, tvPeriod, tvLevel, tvSales;
            TextView tvEdit, tvToggleStatus, tvDelete;

            ViewHolder(View itemView) {
                super(itemView);
                tvName = itemView.findViewById(R.id.tv_name);
                tvStatus = itemView.findViewById(R.id.tv_status);
                tvPrice = itemView.findViewById(R.id.tv_price);
                tvOriginalPrice = itemView.findViewById(R.id.tv_original_price);
                tvPeriod = itemView.findViewById(R.id.tv_period);
                tvLevel = itemView.findViewById(R.id.tv_level);
                tvSales = itemView.findViewById(R.id.tv_sales);
                tvEdit = itemView.findViewById(R.id.tv_edit);
                tvToggleStatus = itemView.findViewById(R.id.tv_toggle_status);
                tvDelete = itemView.findViewById(R.id.tv_delete);
            }
        }
    }
}
