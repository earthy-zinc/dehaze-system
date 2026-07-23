package com.pei.dehaze.ui.input;

import android.app.Activity;
import androidx.appcompat.app.AlertDialog;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.RadioGroup;
import android.widget.Spinner;
import android.widget.TextView;

import com.google.android.material.button.MaterialButton;
import com.google.android.material.textfield.TextInputEditText;
import com.pei.dehaze.R;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.sdk.model.input_history.InputHistoryForm;
import com.pei.dehaze.sdk.model.input_history.InputHistoryUpdateForm;
import com.pei.dehaze.sdk.model.input_history.InputHistoryVO;
import com.pei.dehaze.sdk.model.input_history.InputSource;
import com.pei.dehaze.sdk.model.input_history.ProcessStatus;
import com.pei.dehaze.utils.StringUtils;
import com.pei.dehaze.utils.ToastUtils;

/**
 * 图像输入历史表单 Dialog，封装新增/编辑表单的 UI 渲染与字段采集。
 * 图片选择与文件上传由 Activity 协调，通过 {@link Callback} 通知。
 */
public class InputHistoryFormDialog {

    private static final ProcessStatus[] STATUS_VALUES = ProcessStatus.values();
    private static final InputSource[] FORM_SOURCE_VALUES = InputSource.values();

    private final Activity activity;
    private final Callback callback;

    private AlertDialog dialog;
    private TextView tvOriginalUrl;

    /** 当前编辑的历史记录ID，0 表示新增模式 */
    private long editingHistoryId;
    /** 当前表单中已上传的原始图片 URL */
    private String currentOriginalUrl;

    public InputHistoryFormDialog(Activity activity, Callback callback) {
        this.activity = activity;
        this.callback = callback;
    }

    /**
     * 显示表单 Dialog
     *
     * @param existing 编辑模式传入既有记录；新增模式传 null
     */
    public void show(InputHistoryVO existing) {
        boolean isEdit = existing != null;
        editingHistoryId = isEdit ? existing.getId() : 0;
        currentOriginalUrl = isEdit ? existing.getOriginalImageUrl() : null;

        View view = LayoutInflater.from(activity).inflate(R.layout.dialog_input_history_form, null);

        MaterialButton btnPickImage = view.findViewById(R.id.btn_pick_image);
        tvOriginalUrl = view.findViewById(R.id.tv_original_url);
        TextInputEditText etAlgorithmId = view.findViewById(R.id.et_algorithm_id);
        TextInputEditText etAlgorithmName = view.findViewById(R.id.et_algorithm_name);
        TextInputEditText etAlgorithmParams = view.findViewById(R.id.et_algorithm_params);
        TextInputEditText etProcessingTime = view.findViewById(R.id.et_processing_time);
        Spinner spinnerStatus = view.findViewById(R.id.spinner_status);
        Spinner spinnerSource = view.findViewById(R.id.spinner_source);
        RadioGroup rgFavorite = view.findViewById(R.id.rg_favorite);

        setupSpinner(spinnerStatus, labelsOf(STATUS_VALUES));
        setupSpinner(spinnerSource, labelsOf(FORM_SOURCE_VALUES));

        if (isEdit) {
            fillForm(existing, tvOriginalUrl, etAlgorithmId, etAlgorithmName,
                    etAlgorithmParams, etProcessingTime, spinnerStatus, spinnerSource, rgFavorite);
        } else {
            spinnerStatus.setSelection(0);
            spinnerSource.setSelection(0);
            rgFavorite.check(R.id.rb_favorite_no);
        }

        btnPickImage.setOnClickListener(v -> {
            if (callback != null) callback.onPickImage();
        });

        dialog = new AlertDialog.Builder(activity)
                .setTitle(isEdit ? "修改历史记录" : "新增历史记录")
                .setView(view)
                .setPositiveButton("确定", (d, which) -> submit(
                        isEdit, etAlgorithmId, etAlgorithmName, etAlgorithmParams,
                        etProcessingTime, spinnerStatus, spinnerSource, rgFavorite))
                .setNegativeButton("取消", null)
                .setOnDismissListener(d -> resetState())
                .show();
    }

    /**
     * 文件上传成功回调（由 Activity 转发）
     */
    public void onFileUploaded(FileInfo fileInfo) {
        currentOriginalUrl = fileInfo.getUrl();
        if (tvOriginalUrl != null) {
            tvOriginalUrl.setVisibility(View.VISIBLE);
            tvOriginalUrl.setText("已上传: " + fileInfo.getUrl());
        }
        ToastUtils.showShort(activity, "图片上传成功");

        if (editingHistoryId > 0 && callback != null) {
            InputHistoryUpdateForm form = new InputHistoryUpdateForm();
            form.setOriginalImageUrl(currentOriginalUrl);
            callback.onUpdate(editingHistoryId, form);
        }
    }

    private void setupSpinner(Spinner spinner, String[] labels) {
        ArrayAdapter<String> adapter = new ArrayAdapter<>(activity,
                android.R.layout.simple_spinner_item, labels);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinner.setAdapter(adapter);
    }

    private void fillForm(InputHistoryVO existing, TextView tvOriginalUrl,
                          TextInputEditText etAlgorithmId, TextInputEditText etAlgorithmName,
                          TextInputEditText etAlgorithmParams, TextInputEditText etProcessingTime,
                          Spinner spinnerStatus, Spinner spinnerSource, RadioGroup rgFavorite) {
        if (!TextUtils.isEmpty(currentOriginalUrl)) {
            tvOriginalUrl.setVisibility(View.VISIBLE);
            tvOriginalUrl.setText("已上传: " + currentOriginalUrl);
        }
        if (existing.getAlgorithmId() != null) {
            etAlgorithmId.setText(String.valueOf(existing.getAlgorithmId()));
        }
        etAlgorithmName.setText(StringUtils.safe(existing.getAlgorithmName()));
        etAlgorithmParams.setText(StringUtils.safe(existing.getAlgorithmParams()));
        if (existing.getProcessingTime() != null) {
            etProcessingTime.setText(String.valueOf(existing.getProcessingTime()));
        }
        spinnerStatus.setSelection(indexOf(STATUS_VALUES, existing.getStatus()));
        spinnerSource.setSelection(indexOf(FORM_SOURCE_VALUES, existing.getInputSource()));
        rgFavorite.check(existing.getIsFavorite() != null && existing.getIsFavorite() == 1
                ? R.id.rb_favorite_yes : R.id.rb_favorite_no);
    }

    private void submit(boolean isEdit, TextInputEditText etAlgorithmId,
                        TextInputEditText etAlgorithmName, TextInputEditText etAlgorithmParams,
                        TextInputEditText etProcessingTime, Spinner spinnerStatus,
                        Spinner spinnerSource, RadioGroup rgFavorite) {
        String algorithmIdStr = StringUtils.getText(etAlgorithmId);
        String algorithmName = StringUtils.getText(etAlgorithmName);
        String algorithmParams = StringUtils.getText(etAlgorithmParams);
        String processingTimeStr = StringUtils.getText(etProcessingTime);
        ProcessStatus status = STATUS_VALUES[spinnerStatus.getSelectedItemPosition()];
        InputSource source = FORM_SOURCE_VALUES[spinnerSource.getSelectedItemPosition()];
        Integer isFavorite = rgFavorite.getCheckedRadioButtonId() == R.id.rb_favorite_yes ? 1 : 0;
        Long algorithmId = parseLongOrNull(algorithmIdStr);
        Integer processingTime = parseIntOrNull(processingTimeStr);

        if (isEdit) {
            InputHistoryUpdateForm form = new InputHistoryUpdateForm();
            form.setOriginalImageUrl(currentOriginalUrl);
            form.setAlgorithmId(algorithmId);
            form.setAlgorithmName(algorithmName);
            form.setAlgorithmParams(algorithmParams);
            form.setProcessingTime(processingTime);
            form.setStatus(status);
            form.setIsFavorite(isFavorite);
            callback.onUpdate(editingHistoryId, form);
        } else {
            if (TextUtils.isEmpty(currentOriginalUrl)) {
                ToastUtils.showShort(activity, "请先选择并上传原始图片");
                return;
            }
            InputHistoryForm form = new InputHistoryForm();
            form.setOriginalImageUrl(currentOriginalUrl);
            form.setAlgorithmId(algorithmId);
            form.setAlgorithmName(algorithmName);
            form.setAlgorithmParams(algorithmParams);
            form.setProcessingTime(processingTime);
            form.setStatus(status);
            form.setInputSource(source);
            form.setIsFavorite(isFavorite);
            callback.onCreate(form);
        }
    }

    private void resetState() {
        editingHistoryId = 0;
        currentOriginalUrl = null;
        tvOriginalUrl = null;
        dialog = null;
    }

    private static String[] labelsOf(ProcessStatus[] values) {
        String[] labels = new String[values.length];
        for (int i = 0; i < values.length; i++) {
            labels[i] = values[i].getLabel();
        }
        return labels;
    }

    private static String[] labelsOf(InputSource[] values) {
        String[] labels = new String[values.length];
        for (int i = 0; i < values.length; i++) {
            labels[i] = values[i].getLabel();
        }
        return labels;
    }

    private static int indexOf(Object[] array, Object target) {
        if (target == null) return 0;
        for (int i = 0; i < array.length; i++) {
            if (array[i] != null && array[i].equals(target)) return i;
        }
        return 0;
    }

    private static Long parseLongOrNull(String s) {
        if (TextUtils.isEmpty(s)) return null;
        try {
            return Long.parseLong(s);
        } catch (NumberFormatException e) {
            return null;
        }
    }

    private static Integer parseIntOrNull(String s) {
        if (TextUtils.isEmpty(s)) return null;
        try {
            return Integer.parseInt(s);
        } catch (NumberFormatException e) {
            return null;
        }
    }

    /** 表单提交与图片选择回调 */
    public interface Callback {
        /** 用户点击选择图片 */
        void onPickImage();

        /** 新增模式提交 */
        void onCreate(InputHistoryForm form);

        /** 编辑模式提交 */
        void onUpdate(long historyId, InputHistoryUpdateForm form);
    }
}
