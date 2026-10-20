package com.pei.clarity.ui.metrics_manage.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.clarity.repository.RepositoryAdapters;
import com.pei.clarity.sdk.api.ModelAPI;
import com.pei.clarity.sdk.model.evaluation.EvaluationLogVO;
import com.pei.clarity.ui.common.BaseViewModel;

import java.util.ArrayList;
import java.util.List;

/**
 * 指标管理 ViewModel
 */
public class MetricsManageViewModel extends BaseViewModel {

    private final MutableLiveData<List<EvaluationLogVO>> evalLogs = new MutableLiveData<>();

    public LiveData<List<EvaluationLogVO>> getEvalLogs() {
        return evalLogs;
    }

    public void loadEvalLogs(String modelId) {
        ModelAPI.listEvaluationLogs(modelId, 1, 50,
                RepositoryAdapters.wrap(withLoading(data -> {
                    List<EvaluationLogVO> records = data != null ? data.getList() : null;
                    evalLogs.postValue(records != null ? records : new ArrayList<>());
                })));
    }
}
