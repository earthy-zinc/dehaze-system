package com.pei.dehaze.ui.system;

import android.os.Bundle;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import androidx.swiperefreshlayout.widget.SwipeRefreshLayout;

import com.google.android.material.appbar.MaterialToolbar;
import com.google.android.material.button.MaterialButton;
import com.google.android.material.textfield.TextInputEditText;
import com.pei.dehaze.R;
import com.pei.dehaze.ui.system.adapter.DictItemAdapter;
import com.pei.dehaze.ui.system.viewmodel.DictItemViewModel;
import com.pei.dehaze.sdk.model.dict.DictForm;
import com.pei.dehaze.sdk.model.dict.DictPageVO;

public class DictItemListActivity extends AppCompatActivity {

    private DictItemViewModel dictItemViewModel;
    private DictItemAdapter dictItemAdapter;
    private SwipeRefreshLayout swipeRefreshLayout;
    private TextView tvPageInfo;
    private String typeCode;
    private String typeName;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_dict_item_list);

        typeCode = getIntent().getStringExtra(DictTypeListActivity.EXTRA_TYPE_CODE);
        typeName = getIntent().getStringExtra(DictTypeListActivity.EXTRA_TYPE_NAME);

        initViews();
        initViewModel();
        setupObservers();
        loadData();
    }

    private void initViews() {
        MaterialToolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);
        if (getSupportActionBar() != null) {
            getSupportActionBar().setDisplayHomeAsUpEnabled(true);
        }
        toolbar.setNavigationOnClickListener(v -> finish());
        if (!TextUtils.isEmpty(typeName)) {
            getSupportActionBar().setTitle("字典数据 - " + typeName);
        }

        RecyclerView recyclerView = findViewById(R.id.recycler_view);
        swipeRefreshLayout = findViewById(R.id.swipe_refresh);
        tvPageInfo = findViewById(R.id.tv_page_info);

        dictItemAdapter = new DictItemAdapter();
        dictItemAdapter.setListener(new DictItemAdapter.OnDictItemActionListener() {
            @Override
            public void onEdit(DictPageVO dict) {
                dictItemViewModel.loadDictForm(dict.getId());
            }

            @Override
            public void onDelete(DictPageVO dict) {
                confirmDelete(dict);
            }
        });
        recyclerView.setLayoutManager(new LinearLayoutManager(this));
        recyclerView.setAdapter(dictItemAdapter);

        swipeRefreshLayout.setOnRefreshListener(() -> dictItemViewModel.loadDicts(typeCode));

        MaterialButton btnAdd = findViewById(R.id.btn_add);
        MaterialButton btnPrev = findViewById(R.id.btn_prev);
        MaterialButton btnNext = findViewById(R.id.btn_next);

        btnAdd.setOnClickListener(v -> showFormDialog(null));

        btnPrev.setOnClickListener(v -> {
            int page = dictItemViewModel.getCurrentPage();
            if (page > 1) {
                dictItemViewModel.loadPage(page - 1);
            } else {
                Toast.makeText(this, "已经是第一页", Toast.LENGTH_SHORT).show();
            }
        });

        btnNext.setOnClickListener(v -> {
            int page = dictItemViewModel.getCurrentPage();
            dictItemViewModel.loadPage(page + 1);
        });
    }

    private void initViewModel() {
        dictItemViewModel = new DictItemViewModel();
    }

    private void setupObservers() {
        dictItemViewModel.getDictList().observe(this, list ->
                dictItemAdapter.submitList(list));

        dictItemViewModel.getTotal().observe(this, total -> {
            int page = dictItemViewModel.getCurrentPage();
            int size = dictItemViewModel.getPageSize();
            long totalLong = total != null ? total : 0;
            int totalPages = (int) Math.ceil((double) totalLong / size);
            tvPageInfo.setText(String.format("第 %d 页 / 共 %d 页 (总计 %d 条)", page, totalPages, totalLong));
        });

        dictItemViewModel.getLoading().observe(this, isLoading ->
                swipeRefreshLayout.setRefreshing(isLoading != null && isLoading));

        dictItemViewModel.getError().observe(this, errorMsg -> {
            if (!TextUtils.isEmpty(errorMsg)) {
                Toast.makeText(this, errorMsg, Toast.LENGTH_SHORT).show();
            }
        });

        dictItemViewModel.getActionResult().observe(this, result -> {
            if (!TextUtils.isEmpty(result)) {
                Toast.makeText(this, result, Toast.LENGTH_SHORT).show();
            }
        });

        dictItemViewModel.getDictForm().observe(this, form -> showFormDialog(form));
    }

    private void loadData() {
        dictItemViewModel.loadDicts(typeCode);
    }

    private void confirmDelete(DictPageVO dict) {
        new AlertDialog.Builder(this)
                .setTitle("确认删除")
                .setMessage("确认删除字典数据「" + dict.getName() + "」吗？")
                .setPositiveButton("确认", (dialog, which) -> dictItemViewModel.deleteDict(dict.getId()))
                .setNegativeButton("取消", null)
                .show();
    }

    private void showFormDialog(DictForm existingForm) {
        boolean isEdit = existingForm != null;
        View formView = LayoutInflater.from(this).inflate(R.layout.dialog_dict_item_form, null);

        TextInputEditText etName = formView.findViewById(R.id.et_name);
        TextInputEditText etValue = formView.findViewById(R.id.et_value);
        TextInputEditText etSort = formView.findViewById(R.id.et_sort);
        TextInputEditText etRemark = formView.findViewById(R.id.et_remark);
        RadioGroup rgStatus = formView.findViewById(R.id.rg_status);

        if (isEdit) {
            etName.setText(existingForm.getName());
            etValue.setText(existingForm.getValue());
            etSort.setText(existingForm.getSort() != null ? String.valueOf(existingForm.getSort()) : "1");
            etRemark.setText(existingForm.getRemark());
            if (existingForm.getStatus() != null && existingForm.getStatus() == 1) {
                ((RadioButton) rgStatus.findViewById(R.id.rb_status_enable)).setChecked(true);
            } else {
                ((RadioButton) rgStatus.findViewById(R.id.rb_status_disable)).setChecked(true);
            }
        } else {
            etSort.setText("1");
        }

        AlertDialog dialog = new AlertDialog.Builder(this)
                .setTitle(isEdit ? "修改字典数据" : "新增字典数据")
                .setView(formView)
                .setPositiveButton("确定", null)
                .setNegativeButton("取消", null)
                .create();

        dialog.setOnShowListener(d -> dialog.getButton(AlertDialog.BUTTON_POSITIVE)
                .setOnClickListener(v -> {
                    String name = getText(etName);
                    if (TextUtils.isEmpty(name)) {
                        Toast.makeText(this, "字典标签不能为空", Toast.LENGTH_SHORT).show();
                        return;
                    }
                    String value = getText(etValue);
                    if (TextUtils.isEmpty(value)) {
                        Toast.makeText(this, "字典键值不能为空", Toast.LENGTH_SHORT).show();
                        return;
                    }
                    String sortStr = getText(etSort);
                    if (TextUtils.isEmpty(sortStr)) {
                        Toast.makeText(this, "排序不能为空", Toast.LENGTH_SHORT).show();
                        return;
                    }

                    DictForm form = new DictForm();
                    if (isEdit) {
                        form.setId(existingForm.getId());
                    }
                    form.setName(name);
                    form.setValue(value);
                    form.setSort(Integer.parseInt(sortStr));
                    form.setStatus(rgStatus.getCheckedRadioButtonId() == R.id.rb_status_enable ? 1 : 0);
                    form.setRemark(getText(etRemark));
                    form.setTypeCode(typeCode);

                    if (isEdit) {
                        dictItemViewModel.updateDict(existingForm.getId(), form);
                    } else {
                        dictItemViewModel.addDict(form);
                    }
                    dialog.dismiss();
                }));

        dialog.show();
    }

    private String getText(TextInputEditText et) {
        return et.getText() != null ? et.getText().toString().trim() : "";
    }
}
