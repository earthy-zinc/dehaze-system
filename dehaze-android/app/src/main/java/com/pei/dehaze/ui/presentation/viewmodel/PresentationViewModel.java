package com.pei.dehaze.ui.presentation.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.sdk.api.AlgorithmAPI;
import com.pei.dehaze.sdk.api.FileAPI;
import com.pei.dehaze.sdk.api.ModelAPI;
import com.pei.dehaze.ui.common.BaseViewModel;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmQuery;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.sdk.model.prediction.DehazeParams;
import com.pei.dehaze.sdk.model.prediction.PredEvalTaskStatus;
import com.pei.dehaze.sdk.model.prediction.PredParam;
import com.pei.dehaze.sdk.model.prediction.PredResult;
import com.pei.dehaze.sdk.model.prediction.PredictionLogVO;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class PresentationViewModel extends BaseViewModel {

    private final MutableLiveData<FileInfo> uploadedFile = new MutableLiveData<>();
    private final MutableLiveData<List<Option>> algorithmOptions = new MutableLiveData<>();
    private final MutableLiveData<Algorithm> algorithmDetail = new MutableLiveData<>();
    private final MutableLiveData<PredResult> predictionResult = new MutableLiveData<>();
    private final MutableLiveData<List<PredictionLogVO>> historyList = new MutableLiveData<>();

    private String originalImageUrl;

    public void uploadImage(File imageFile) {
        FileAPI.upload(imageFile, RepositoryAdapters.wrap(withLoading(fileInfo -> {
            uploadedFile.postValue(fileInfo);
            originalImageUrl = fileInfo.getUrl();
            operationResult.postValue("图片上传成功");
        })));
    }

    public void loadAlgorithmOptions() {
        AlgorithmAPI.getOption(RepositoryAdapters.wrap(withLoading(algorithmOptions::postValue)));
    }

    public void loadAlgorithms(AlgorithmQuery query) {
        AlgorithmAPI.getList(query, RepositoryAdapters.wrap(withLoading(algorithms -> { })));
    }

    public void getAlgorithmDetail(long id) {
        AlgorithmAPI.getAlgorithmInfoById(id, RepositoryAdapters.wrap(withLoading(algorithmDetail::postValue)));
    }

    public void predict(long algorithmId, DehazeParams params) {
        if (originalImageUrl == null) {
            error.setValue("请先上传图片");
            return;
        }
        String invalidMsg = params == null ? null : params.validate();
        if (invalidMsg != null) {
            error.setValue(invalidMsg);
            return;
        }
        PredParam param = new PredParam();
        param.setAlgorithmId(algorithmId);
        param.setImageUrl(originalImageUrl);
        param.setParams(params);
        ModelAPI.predictAndWait(param, RepositoryAdapters.wrap(withLoading(result -> {
            if (result.getStatus() == PredEvalTaskStatus.FAILED) {
                error.postValue(result.getErrorMessage() != null ? result.getErrorMessage() : "去雾处理失败");
                return;
            }
            predictionResult.postValue(result);
            operationResult.postValue("去雾处理完成");
            loadHistory();
        })));
    }

    public void loadHistory() {
        ModelAPI.listPredictionLogs(null, 1, 20, RepositoryAdapters.wrapPage(withLoading(logs ->
                historyList.postValue(logs != null ? logs : new ArrayList<>()))));
    }

    public LiveData<FileInfo> getUploadedFile() {
        return uploadedFile;
    }

    public LiveData<List<Option>> getAlgorithmOptions() {
        return algorithmOptions;
    }

    public LiveData<Algorithm> getAlgorithmDetail() {
        return algorithmDetail;
    }

    public LiveData<PredResult> getPredictionResult() {
        return predictionResult;
    }

    public LiveData<List<PredictionLogVO>> getHistoryList() {
        return historyList;
    }

    public String getOriginalImageUrl() {
        return originalImageUrl;
    }
}
