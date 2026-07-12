package com.pei.dehaze.ui.evaluation.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.pei.dehaze.repository.EvaluationRepository;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.sdk.model.evaluation.EvalParam;
import com.pei.dehaze.sdk.model.evaluation.EvalResult;
import com.pei.dehaze.sdk.model.evaluation.EvaluationLogVO;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.sdk.model.prediction.PredParam;
import com.pei.dehaze.sdk.model.prediction.PredResult;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class EvaluationViewModel extends ViewModel {

    private final EvaluationRepository evaluationRepository;

    private final MutableLiveData<FileInfo> hazyFile = new MutableLiveData<>();
    private final MutableLiveData<FileInfo> clearFile = new MutableLiveData<>();
    private final MutableLiveData<List<Option>> algorithmOptions = new MutableLiveData<>();
    private final MutableLiveData<PredResult> predictionResult = new MutableLiveData<>();
    private final MutableLiveData<EvalResult> evaluationResult = new MutableLiveData<>();
    private final MutableLiveData<Algorithm> algorithmInfo = new MutableLiveData<>();
    private final MutableLiveData<List<EvaluationLogVO>> evaluationLogs = new MutableLiveData<>();
    private final MutableLiveData<Boolean> loading = new MutableLiveData<>(false);
    private final MutableLiveData<String> error = new MutableLiveData<>();
    private final MutableLiveData<String> operationResult = new MutableLiveData<>();

    public EvaluationViewModel() {
        evaluationRepository = new EvaluationRepository();
    }

    public void uploadHazyImage(File imageFile) {
        loading.setValue(true);
        evaluationRepository.uploadImage(imageFile, new EvaluationRepository.UploadCallback() {
            @Override
            public void onSuccess(FileInfo fileInfo) {
                hazyFile.postValue(fileInfo);
                operationResult.postValue("有雾图上传成功");
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void uploadClearImage(File imageFile) {
        loading.setValue(true);
        evaluationRepository.uploadImage(imageFile, new EvaluationRepository.UploadCallback() {
            @Override
            public void onSuccess(FileInfo fileInfo) {
                clearFile.postValue(fileInfo);
                operationResult.postValue("清晰图上传成功");
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void loadAlgorithmOptions() {
        loading.setValue(true);
        evaluationRepository.getAlgorithmOptions(new EvaluationRepository.OptionsCallback() {
            @Override
            public void onSuccess(List<Option> options) {
                algorithmOptions.postValue(options);
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
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
        loading.setValue(true);
        evaluationRepository.getPrediction(param, new EvaluationRepository.PredictionCallback() {
            @Override
            public void onSuccess(PredResult result) {
                predictionResult.postValue(result);
                operationResult.postValue("去雾处理完成，可进行评估");
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
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
        loading.setValue(true);
        evaluationRepository.getEvaluation(param, new EvaluationRepository.EvaluationCallback() {
            @Override
            public void onSuccess(EvalResult result) {
                evaluationResult.postValue(result);
                operationResult.postValue("评估完成");
                loading.postValue(false);
                loadEvaluationLogs();
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void getAlgorithmInfo(int id) {
        loading.setValue(true);
        evaluationRepository.getAlgorithmInfo(id, new EvaluationRepository.AlgorithmCallback() {
            @Override
            public void onSuccess(Algorithm algorithm) {
                algorithmInfo.postValue(algorithm);
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void loadEvaluationLogs() {
        evaluationRepository.listEvaluationLogs(1, 20, new EvaluationRepository.EvaluationLogListCallback() {
            @Override
            public void onSuccess(List<EvaluationLogVO> logs) {
                evaluationLogs.postValue(logs != null ? logs : new ArrayList<>());
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
            }
        });
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

    public LiveData<Boolean> getLoading() {
        return loading;
    }

    public LiveData<String> getError() {
        return error;
    }

    public LiveData<String> getOperationResult() {
        return operationResult;
    }

    public void clearError() {
        error.setValue(null);
    }

    public void clearOperationResult() {
        operationResult.setValue(null);
    }
}
