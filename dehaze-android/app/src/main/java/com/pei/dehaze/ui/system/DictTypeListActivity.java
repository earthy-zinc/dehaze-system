package com.pei.dehaze.ui.system;

import android.content.Intent;
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
import com.pei.dehaze.ui.system.adapter.DictTypeAdapter;
import com.pei.dehaze.ui.system.viewmodel.DictTypeViewModel;
import com.pei.dehaze.sdk.model.dict.DictTypeForm;
import com.pei.dehaze.sdk.model.dict.DictTypePageVO;
import com.pei.dehaze.utils.StringUtils;

import java.util.Collections;

public class DictTypeListActivity extends AppCompatActivity {

    public static final String EXTRA_TYPE_CODE = "type_code";
    public static final String EXTRA_TYPE_NAME = "type_name";

    private DictTypeViewModel dictTypeViewModel;
    private DictTypeAdapter dictTypeAdapter;
    private SwipeRefreshLayout swipeRefreshLayout;
    private TextInputEditText etSearch;
    private TextView tvPageInfo;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_dict_type_list);

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

        etSearch = findViewById(R.id.et_search);
        RecyclerView recyclerView = findViewById(R.id.recycler_view);
        swipeRefreshLayout = findViewById(R.id.swipe_refresh);
        tvPageInfo = findViewById(R.id.tv_page_info);

        dictTypeAdapter = new DictTypeAdapter();
        dictTypeAdapter.setListener(new DictTypeAdapter.OnDictTypeActionListener() {
            @Override
            public void onEdit(DictTypePageVO dictType) {
                dictTypeViewModel.loadDictTypeForm(dictType.getId());
            }

            @Override
            public void onDelete(DictTypePageVO dictType) {
                confirmDelete(dictType);
            }

            @Override
            public void onManageItems(DictTypePageVO dictType) {
                Intent intent = new Intent(DictTypeListActivity.this, DictItemListActivity.class);
                intent.putExtra(EXTRA_TYPE_CODE, dictType.getCode());
                intent.putExtra(EXTRA_TYPE_NAME, dictType.getName());
                startActivity(intent);
            }
        });
        recyclerView.setLayoutManager(new LinearLayoutManager(this));
        recyclerView.setAdapter(dictTypeAdapter);

        swipeRefreshLayout.setOnRefreshListener(() -> dictTypeViewModel.loadDictTypes(null));

        MaterialButton btnSearch = findViewById(R.id.btn_search);
        MaterialButton btnReset = findViewById(R.id.btn_reset);
        MaterialButton btnAdd = findViewById(R.id.btn_add);
        MaterialButton btnPrev = findViewById(R.id.btn_prev);
        MaterialButton btnNext = findViewById(R.id.btn_next);

        btnSearch.setOnClickListener(v -> {
            String keywords = StringUtils.getText(etSearch);
            dictTypeViewModel.loadDictTypes(keywords.isEmpty() ? null : keywords);
        });

        btnReset.setOnClickListener(v -> {
            etSearch.setText("");
            dictTypeViewModel.loadDictTypes(null);
        });

        btnAdd.setOnClickListener(v -> showFormDialog(null));

        btnPrev.setOnClickListener(v -> {
            int page = dictTypeViewModel.getCurrentPage();
            if (page > 1) {
                dictTypeViewModel.loadPage(page - 1);
            } else {
                Toast.makeText(this, "已经是第一页", Toast.LENGTH_SHORT).show();
            }
        });

        btnNext.setOnClickListener(v -> {
            int page = dictTypeViewModel.getCurrentPage();
            dictTypeViewModel.loadPage(page + 1);
        });
    }

    private void initViewModel() {
        dictTypeViewModel = new DictTypeViewModel();
    }

    private void setupObservers() {
        dictTypeViewModel.getDictTypeList().observe(this, list ->
                dictTypeAdapter.submitList(list));

        dictTypeViewModel.getTotal().observe(this, total -> {
            int page = dictTypeViewModel.getCurrentPage();
            int size = dictTypeViewModel.getPageSize();
            long totalLong = total != null ? total : 0;
            int totalPages = (int) Math.ceil((double) totalLong / size);
            tvPageInfo.setText(String.format("第 %d 页 / 共 %d 页 (总计 %d 条)", page, totalPages, totalLong));
        });

        dictTypeViewModel.getLoading().observe(this, isLoading ->
                swipeRefreshLayout.setRefreshing(isLoading != null && isLoading));

        dictTypeViewModel.getError().observe(this, errorMsg -> {
            if (!TextUtils.isEmpty(errorMsg)) {
                Toast.makeText(this, errorMsg, Toast.LENGTH_SHORT).show();
            }
        });

        dictTypeViewModel.getOperationResult().observe(this, result -> {
            if (!TextUtils.isEmpty(result)) {
                Toast.makeText(this, result, Toast.LENGTH_SHORT).show();
            }
        });

        dictTypeViewModel.getDictTypeForm().observe(this, form -> showFormDialog(form));
    }

    private void loadData() {
        dictTypeViewModel.loadDictTypes(null);
    }

    private void confirmDelete(DictTypePageVO dictType) {
        new AlertDialog.Builder(this)
                .setTitle("确认删除")
                .setMessage("确认删除字典类型「" + dictType.getName() + "」吗？")
                .setPositiveButton("确认", (dialog, which) -> dictTypeViewModel.deleteDictType(Collections.singletonList((long) dictType.getId())))
                .setNegativeButton("取消", null)
                .show();
    }

    private void showFormDialog(DictTypeForm existingForm) {
        boolean isEdit = existingForm != null;
        View formView = LayoutInflater.from(this).inflate(R.layout.dialog_dict_type_form, null);

        TextInputEditText etName = formView.findViewById(R.id.et_name);
        TextInputEditText etCode = formView.findViewById(R.id.et_code);
        TextInputEditText etRemark = formView.findViewById(R.id.et_remark);
        RadioGroup rgStatus = formView.findViewById(R.id.rg_status);

        if (isEdit) {
            etName.setText(existingForm.getName());
            etCode.setText(existingForm.getCode());
            etRemark.setText(existingForm.getRemark());
            if (existingForm.getStatus() == 1) {
                ((RadioButton) rgStatus.findViewById(R.id.rb_status_enable)).setChecked(true);
            } else {
                ((RadioButton) rgStatus.findViewById(R.id.rb_status_disable)).setChecked(true);
            }
        }

        AlertDialog dialog = new AlertDialog.Builder(this)
                .setTitle(isEdit ? "修改字典类型" : "新增字典类型")
                .setView(formView)
                .setPositiveButton("确定", null)
                .setNegativeButton("取消", null)
                .create();

        dialog.setOnShowListener(d -> dialog.getButton(AlertDialog.BUTTON_POSITIVE)
                .setOnClickListener(v -> {
                    String name = StringUtils.getText(etName);
                    if (TextUtils.isEmpty(name)) {
                        Toast.makeText(this, "字典名称不能为空", Toast.LENGTH_SHORT).show();
                        return;
                    }
                    String code = StringUtils.getText(etCode);
                    if (TextUtils.isEmpty(code)) {
                        Toast.makeText(this, "字典编码不能为空", Toast.LENGTH_SHORT).show();
                        return;
                    }

                    DictTypeForm form = new DictTypeForm();
                    if (isEdit) {
                        form.setId(existingForm.getId());
                    }
                    form.setName(name);
                    form.setCode(code);
                    form.setRemark(StringUtils.getText(etRemark));
                    form.setStatus(rgStatus.getCheckedRadioButtonId() == R.id.rb_status_enable ? 1 : 0);

                    if (isEdit) {
                        dictTypeViewModel.updateDictType(existingForm.getId(), form);
                    } else {
                        dictTypeViewModel.addDictType(form);
                    }
                    dialog.dismiss();
                }));

        dialog.show();
    }
}
