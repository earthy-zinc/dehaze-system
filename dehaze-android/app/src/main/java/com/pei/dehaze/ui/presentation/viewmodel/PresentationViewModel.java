package com.pei.dehaze.ui.presentation.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.AlgorithmRepository;
import com.pei.dehaze.repository.SharedRepository;
import com.pei.dehaze.ui.common.BaseViewModel;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmQuery;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.sdk.model.prediction.DehazeParams;
import com.pei.dehaze.sdk.model.prediction.PredParam;
import com.pei.dehaze.sdk.model.prediction.PredResult;
import com.pei.dehaze.sdk.model.prediction.PredictionLogVO;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class PresentationViewModel extends BaseViewModel {

    private final SharedRepository sharedRepository;
    private final AlgorithmRepository algorithmRepository;

    private final MutableLiveData<FileInfo> uploadedFile = new MutableLiveData<>();
    private final MutableLiveData<List<Option>> algorithmOptions = new MutableLiveData<>();
    private final MutableLiveData<Algorithm> algorithmDetail = new MutableLiveData<>();
    private final MutableLiveData<PredResult> predictionResult = new MutableLiveData<>();
    private final MutableLiveData<List<PredictionLogVO>> historyList = new MutableLiveData<>();

    private String originalImageUrl;

    public PresentationViewModel() {
        sharedRepository = new SharedRepository();
        algorithmRepository = new AlgorithmRepository();
    }

    public void uploadImage(File imageFile) {
        sharedRepository.uploadImage(imageFile, withLoading(fileInfo -> {
            uploadedFile.postValue(fileInfo);
            originalImageUrl = fileInfo.getUrl();
            operationResult.postValue("图片上传成功");
        }));
    }

    public void loadAlgorithmOptions() {
        sharedRepository.getAlgorithmOptions(withLoading(algorithmOptions::postValue));
    }

    public void loadAlgorithms(AlgorithmQuery query) {
        algorithmRepository.getAlgorithms(query, withLoading(algorithms -> { }));
    }

    public void getAlgorithmDetail(long id) {
        algorithmRepository.getAlgorithmDetail(id, withLoading(algorithmDetail::postValue));
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
        sharedRepository.getPrediction(param, withLoading(result -> {
            predictionResult.postValue(result);
            operationResult.postValue("去雾处理完成");
            loadHistory();
        }));
    }

    public void loadHistory() {
        sharedRepository.listPredictionLogs(1, 20, withLoading(logs ->
                historyList.postValue(logs != null ? logs : new ArrayList<>())));
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
