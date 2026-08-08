package com.pei.dehaze.ui.metrics_manage.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.repository.RepositoryCallback;
import com.pei.dehaze.sdk.api.ModelAPI;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.evaluation.EvaluationLogVO;
import com.pei.dehaze.sdk.model.prediction.PredictionLogVO;
import com.pei.dehaze.ui.common.BaseViewModel;

import java.util.ArrayList;
import java.util.List;

/**
 * 指标管理 ViewModel
 */
public class MetricsManageViewModel extends BaseViewModel {

    private final MutableLiveData<List<EvaluationLogVO>> evalLogs = new MutableLiveData<>();
    private final MutableLiveData<List<PredictionLogVO>> predLogs = new MutableLiveData<>();

    public LiveData<List<EvaluationLogVO>> getEvalLogs() {
        return evalLogs;
    }

    public LiveData<List<PredictionLogVO>> getPredLogs() {
        return predLogs;
    }

    public void loadEvalLogs(Long algorithmId) {
        loading.setValue(true);
        ModelAPI.listEvaluationLogs(algorithmId, 1, 50,
                RepositoryAdapters.wrap(new RepositoryCallback<PageResult<EvaluationLogVO>>() {
                    @Override
                    public void onSuccess(PageResult<EvaluationLogVO> data) {
                        loading.setValue(false);
                        List<EvaluationLogVO> records = data != null ? data.getList() : null;
                        evalLogs.postValue(records != null ? records : new ArrayList<>());
                    }

                    @Override
                    public void onError(String errorMessage) {
                        error.postValue(errorMessage);
                        loading.setValue(false);
                    }
                }));
    }

    public void loadPredLogs(Long algorithmId) {
        loading.setValue(true);
        ModelAPI.listPredictionLogs(algorithmId, 1, 50,
                RepositoryAdapters.wrap(new RepositoryCallback<PageResult<PredictionLogVO>>() {
                    @Override
                    public void onSuccess(PageResult<PredictionLogVO> data) {
                        loading.setValue(false);
                        List<PredictionLogVO> records = data != null ? data.getList() : null;
                        predLogs.postValue(records != null ? records : new ArrayList<>());
                    }

                    @Override
                    public void onError(String errorMessage) {
                        error.postValue(errorMessage);
                        loading.setValue(false);
                    }
                }));
    }
}
