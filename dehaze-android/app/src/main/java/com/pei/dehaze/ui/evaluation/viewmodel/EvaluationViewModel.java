package com.pei.dehaze.ui.evaluation.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.AlgorithmRepository;
import com.pei.dehaze.repository.SharedRepository;
import com.pei.dehaze.ui.common.BaseViewModel;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.sdk.model.evaluation.EvalParam;
import com.pei.dehaze.sdk.model.evaluation.EvalResult;
import com.pei.dehaze.sdk.model.evaluation.EvaluationLogVO;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.sdk.model.prediction.DehazeParams;
import com.pei.dehaze.sdk.model.prediction.PredParam;
import com.pei.dehaze.sdk.model.prediction.PredResult;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class EvaluationViewModel extends BaseViewModel {

    private final SharedRepository sharedRepository;
    private final AlgorithmRepository algorithmRepository;

    private final MutableLiveData<FileInfo> hazyFile = new MutableLiveData<>();
    private final MutableLiveData<FileInfo> clearFile = new MutableLiveData<>();
    private final MutableLiveData<List<Option>> algorithmOptions = new MutableLiveData<>();
    private final MutableLiveData<PredResult> predictionResult = new MutableLiveData<>();
    private final MutableLiveData<EvalResult> evaluationResult = new MutableLiveData<>();
    private final MutableLiveData<Algorithm> algorithmInfo = new MutableLiveData<>();
    private final MutableLiveData<List<EvaluationLogVO>> evaluationLogs = new MutableLiveData<>();

    public EvaluationViewModel() {
        sharedRepository = new SharedRepository();
        algorithmRepository = new AlgorithmRepository();
    }

    public void uploadHazyImage(File imageFile) {
        sharedRepository.uploadImage(imageFile, withLoading(fileInfo -> {
            hazyFile.postValue(fileInfo);
            operationResult.postValue("有雾图上传成功");
        }));
    }

    public void uploadClearImage(File imageFile) {
        sharedRepository.uploadImage(imageFile, withLoading(fileInfo -> {
            clearFile.postValue(fileInfo);
            operationResult.postValue("清晰图上传成功");
        }));
    }

    public void loadAlgorithmOptions() {
        sharedRepository.getAlgorithmOptions(withLoading(algorithmOptions::postValue));
    }

    public void predict(long algorithmId) {
        FileInfo hazy = hazyFile.getValue();
        if (hazy == null || hazy.getUrl() == null) {
            error.setValue("请先上传有雾图片");
            return;
        }
        PredParam param = new PredParam();
        param.setAlgorithmId(algorithmId);
        param.setImageUrl(hazy.getUrl());
        param.setParams(new DehazeParams());
        sharedRepository.getPrediction(param, withLoading(result -> {
            predictionResult.postValue(result);
            operationResult.postValue("去雾处理完成，可进行评估");
        }));
    }

    public void evaluate(long algorithmId) {
        PredResult pred = predictionResult.getValue();
        FileInfo clear = clearFile.getValue();
        if (pred == null || pred.getResultUrl() == null) {
            error.setValue("请先执行去雾处理");
            return;
        }
        if (clear == null || clear.getUrl() == null) {
            error.setValue("请先上传清晰参考图");
            return;
        }
        EvalParam param = new EvalParam();
        param.setAlgorithmId(algorithmId);
        param.setPredUrl(pred.getResultUrl());
        param.setGtUrl(clear.getUrl());
        sharedRepository.getEvaluation(param, withLoading(result -> {
            evaluationResult.postValue(result);
            operationResult.postValue("评估完成");
            loadEvaluationLogs();
        }));
    }

    public void getAlgorithmInfo(int id) {
        algorithmRepository.getAlgorithmDetail(id, withLoading(algorithmInfo::postValue));
    }

    public void loadEvaluationLogs() {
        sharedRepository.listEvaluationLogs(1, 20, withLoading(logs ->
                evaluationLogs.postValue(logs != null ? logs : new ArrayList<>())));
    }

    public LiveData<FileInfo> getHazyFile() {
        return hazyFile;
    }

    public LiveData<FileInfo> getClearFile() {
        return clearFile;
    }

    public LiveData<List<Option>> getAlgorithmOptions() {
        return algorithmOptions;
    }

    public LiveData<PredResult> getPredictionResult() {
        return predictionResult;
    }

    public LiveData<EvalResult> getEvaluationResult() {
        return evaluationResult;
    }

    public LiveData<Algorithm> getAlgorithmInfo() {
        return algorithmInfo;
    }

    public LiveData<List<EvaluationLogVO>> getEvaluationLogs() {
        return evaluationLogs;
    }
}
