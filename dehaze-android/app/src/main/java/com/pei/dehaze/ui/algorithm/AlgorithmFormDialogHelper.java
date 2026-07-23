package com.pei.dehaze.ui.algorithm;

import android.app.Activity;
import android.app.AlertDialog;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.EditText;
import android.widget.Spinner;

import com.pei.dehaze.R;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmStatus;
import com.pei.dehaze.utils.StringUtils;
import com.pei.dehaze.utils.ToastUtils;

/**
 * 算法表单 Dialog 帮助类，统一抽取 {@link AlgorithmListActivity} 与
 * {@link AlgorithmDetailActivity} 中重复的算法新增/编辑表单逻辑。
 */
public final class AlgorithmFormDialogHelper {

    private AlgorithmFormDialogHelper() {
    }

    /**
     * 显示算法表单 Dialog
     *
     * @param activity 所属 Activity
     * @param existing 编辑模式传入既有算法；新增模式传 null
     * @param listener 提交回调
     */
    public static void show(Activity activity, Algorithm existing, OnSubmitListener listener) {
        boolean isEdit = existing != null;
        View view = LayoutInflater.from(activity).inflate(R.layout.dialog_algorithm_form, null);

        EditText etName = view.findViewById(R.id.et_name);
        EditText etType = view.findViewById(R.id.et_type);
        EditText etPath = view.findViewById(R.id.et_path);
        EditText etImportPath = view.findViewById(R.id.et_import_path);
        EditText etParams = view.findViewById(R.id.et_params);
        EditText etFlops = view.findViewById(R.id.et_flops);
        EditText etSize = view.findViewById(R.id.et_size);
        Spinner spinnerStatus = view.findViewById(R.id.spinner_status);
        EditText etDescription = view.findViewById(R.id.et_description);

        setupStatusSpinner(activity, spinnerStatus);

        if (isEdit) {
            fillForm(existing, etName, etType, etPath, etImportPath, etParams,
                    etFlops, etSize, etDescription, spinnerStatus);
        } else {
            spinnerStatus.setSelection(0);
        }

        new AlertDialog.Builder(activity)
                .setTitle(isEdit ? "修改算法" : "新增算法")
                .setView(view)
                .setPositiveButton("确定", (d, which) -> submit(activity, isEdit, existing,
                        etName, etType, etPath, etImportPath, etParams, etFlops, etSize,
                        spinnerStatus, etDescription, listener))
                .setNegativeButton("取消", null)
                .show();
    }

    private static void setupStatusSpinner(Activity activity, Spinner spinnerStatus) {
        AlgorithmStatus[] statuses = AlgorithmStatus.values();
        String[] statusLabels = new String[statuses.length];
        for (int i = 0; i < statuses.length; i++) {
            statusLabels[i] = statuses[i].getLabel();
        }
        ArrayAdapter<String> statusAdapter = new ArrayAdapter<>(activity,
                android.R.layout.simple_spinner_item, statusLabels);
        statusAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinnerStatus.setAdapter(statusAdapter);
    }

    private static void fillForm(Algorithm existing, EditText etName, EditText etType,
                                 EditText etPath, EditText etImportPath, EditText etParams,
                                 EditText etFlops, EditText etSize, EditText etDescription,
                                 Spinner spinnerStatus) {
        etName.setText(StringUtils.safe(existing.getName()));
        etType.setText(StringUtils.safe(existing.getType()));
        etPath.setText(StringUtils.safe(existing.getPath()));
        etImportPath.setText(StringUtils.safe(existing.getImportPath()));
        etParams.setText(StringUtils.safe(existing.getParams()));
        etFlops.setText(StringUtils.safe(existing.getFlops()));
        etSize.setText(StringUtils.safe(existing.getSize()));
        etDescription.setText(StringUtils.safe(existing.getDescription()));
        AlgorithmStatus currentStatus = existing.getStatus() != null
                ? existing.getStatus() : AlgorithmStatus.DRAFT;
        AlgorithmStatus[] editStatuses = AlgorithmStatus.values();
        for (int i = 0; i < editStatuses.length; i++) {
            if (editStatuses[i] == currentStatus) {
                spinnerStatus.setSelection(i);
                break;
            }
        }
    }

    private static void submit(Activity activity, boolean isEdit, Algorithm existing,
                               EditText etName, EditText etType, EditText etPath,
                               EditText etImportPath, EditText etParams, EditText etFlops,
                               EditText etSize, Spinner spinnerStatus, EditText etDescription,
                               OnSubmitListener listener) {
        String name = etName.getText().toString().trim();
        String type = etType.getText().toString().trim();
        String path = etPath.getText().toString().trim();
        String importPath = etImportPath.getText().toString().trim();
        String params = etParams.getText().toString().trim();
        String flops = etFlops.getText().toString().trim();
        String size = etSize.getText().toString().trim();
        String description = etDescription.getText().toString().trim();
        AlgorithmStatus status = AlgorithmStatus.values()[spinnerStatus.getSelectedItemPosition()];

        String error = validate(name, type, path, importPath);
        if (error != null) {
            ToastUtils.showShort(activity, error);
            return;
        }

        Algorithm data = new Algorithm();
        data.setName(name);
        data.setType(type);
        data.setPath(path);
        data.setImportPath(importPath);
        data.setParams(params);
        data.setFlops(flops);
        data.setSize(size);
        data.setDescription(description);
        data.setStatus(status);

        if (isEdit) {
            data.setId(existing.getId());
            data.setParentId(existing.getParentId());
            listener.onUpdate(data, existing.getId());
        } else {
            data.setParentId(0L);
            listener.onCreate(data);
        }
    }

    private static String validate(String name, String type, String path, String importPath) {
        if (TextUtils.isEmpty(name)) return "请输入算法名称";
        if (TextUtils.isEmpty(type)) return "请输入算法类型";
        if (TextUtils.isEmpty(path)) return "请输入模型文件路径";
        if (TextUtils.isEmpty(importPath)) return "请输入模型导入路径";
        return null;
    }

    /** 提交回调 */
    public interface OnSubmitListener {
        /** 新增模式提交 */
        void onCreate(Algorithm data);

        /** 编辑模式提交 */
        void onUpdate(Algorithm data, long existingId);
    }
}
