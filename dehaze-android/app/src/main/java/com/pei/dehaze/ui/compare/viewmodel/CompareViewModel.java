package com.pei.dehaze.ui.compare.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.RepositoryCallback;
import com.pei.dehaze.repository.SharedRepository;
import com.pei.dehaze.ui.common.BaseViewModel;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.evaluation.EvalParam;
import com.pei.dehaze.sdk.model.evaluation.EvalResult;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.sdk.model.prediction.DehazeParams;
import com.pei.dehaze.sdk.model.prediction.PredParam;
import com.pei.dehaze.sdk.model.prediction.PredResult;
import com.pei.dehaze.sdk.model.prediction.PredictionLogVO;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

public class CompareViewModel extends BaseViewModel {

    private final SharedRepository sharedRepository;

    private final MutableLiveData<FileInfo> uploadedFile = new MutableLiveData<>();
    private final MutableLiveData<List<Option>> algorithmOptions = new MutableLiveData<>();
    private final MutableLiveData<PredResult> predictionResult = new MutableLiveData<>();
    private final MutableLiveData<Map<Long, PredResult>> multiPredictionResults = new MutableLiveData<>();
    private final MutableLiveData<EvalResult> evaluationResult = new MutableLiveData<>();
    private final MutableLiveData<List<PredictionLogVO>> predictionLogs = new MutableLiveData<>();

    private String originalImageUrl;

    public CompareViewModel() {
        sharedRepository = new SharedRepository();
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
        }));
    }

    public void predictMultiple(List<Long> algorithmIds, DehazeParams params) {
        if (originalImageUrl == null) {
            error.setValue("请先上传图片");
            return;
        }
        if (algorithmIds == null || algorithmIds.isEmpty()) {
            error.setValue("请至少选择一个算法");
            return;
        }
        String invalidMsg = params == null ? null : params.validate();
        if (invalidMsg != null) {
            error.setValue(invalidMsg);
            return;
        }
        loading.setValue(true);
        Map<Long, PredResult> results = new HashMap<>();
        AtomicInteger pending = new AtomicInteger(algorithmIds.size());
        for (Long algorithmId : algorithmIds) {
            PredParam param = new PredParam();
            param.setAlgorithmId(algorithmId);
            param.setImageUrl(originalImageUrl);
            param.setParams(params);
            sharedRepository.getPrediction(param, new RepositoryCallback<PredResult>() {
                @Override
                public void onSuccess(PredResult result) {
                    synchronized (results) {
                        results.put(algorithmId, result);
                    }
                    if (pending.decrementAndGet() == 0) {
                        multiPredictionResults.postValue(results);
                        operationResult.postValue("多算法处理完成");
                        loading.postValue(false);
                    }
                }

                @Override
                public void onError(String errorMessage) {
                    if (pending.decrementAndGet() == 0) {
                        multiPredictionResults.postValue(results);
                        loading.postValue(false);
                    }
                    error.postValue(errorMessage);
                }
            });
        }
    }

    public void evaluate(long algorithmId, String predUrl, String gtUrl) {
        if (gtUrl == null || gtUrl.isEmpty()) {
            error.setValue("评估需要提供参考图片");
            return;
        }
        EvalParam param = new EvalParam();
        param.setAlgorithmId(algorithmId);
        param.setPredUrl(predUrl);
        param.setGtUrl(gtUrl);
        sharedRepository.getEvaluation(param, withLoading(result -> {
            evaluationResult.postValue(result);
            operationResult.postValue("评估完成");
        }));
    }

    public void loadPredictionLogs() {
        sharedRepository.listPredictionLogs(1, 20, withLoading(logs ->
                predictionLogs.postValue(logs != null ? logs : new ArrayList<>())));
    }

    public LiveData<FileInfo> getUploadedFile() {
        return uploadedFile;
    }

    public LiveData<List<Option>> getAlgorithmOptions() {
        return algorithmOptions;
    }

    public LiveData<PredResult> getPredictionResult() {
        return predictionResult;
    }

    public LiveData<Map<Long, PredResult>> getMultiPredictionResults() {
        return multiPredictionResults;
    }

    public LiveData<EvalResult> getEvaluationResult() {
        return evaluationResult;
    }

    public LiveData<List<PredictionLogVO>> getPredictionLogs() {
        return predictionLogs;
    }

    public String getOriginalImageUrl() {
        return originalImageUrl;
    }
}