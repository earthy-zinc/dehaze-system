package com.pei.dehaze.ui.compare.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.pei.dehaze.repository.CompareRepository;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.evaluation.EvalParam;
import com.pei.dehaze.sdk.model.evaluation.EvalResult;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.sdk.model.prediction.PredParam;
import com.pei.dehaze.sdk.model.prediction.PredResult;
import com.pei.dehaze.sdk.model.prediction.PredictionLogVO;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class CompareViewModel extends ViewModel {

    private final CompareRepository compareRepository;

    private final MutableLiveData<FileInfo> uploadedFile = new MutableLiveData<>();
    private final MutableLiveData<List<Option>> algorithmOptions = new MutableLiveData<>();
    private final MutableLiveData<PredResult> predictionResult = new MutableLiveData<>();
    private final MutableLiveData<Map<String, PredResult>> multiPredictionResults = new MutableLiveData<>();
    private final MutableLiveData<EvalResult> evaluationResult = new MutableLiveData<>();
    private final MutableLiveData<List<PredictionLogVO>> predictionLogs = new MutableLiveData<>();
    private final MutableLiveData<Boolean> loading = new MutableLiveData<>(false);
    private final MutableLiveData<String> error = new MutableLiveData<>();
    private final MutableLiveData<String> operationResult = new MutableLiveData<>();

    private String originalImageUrl;

    public CompareViewModel() {
        compareRepository = new CompareRepository();
    }

    public void uploadImage(File imageFile) {
        loading.setValue(true);
        compareRepository.uploadImage(imageFile, new CompareRepository.UploadCallback() {
            @Override
            public void onSuccess(FileInfo fileInfo) {
                uploadedFile.postValue(fileInfo);
                originalImageUrl = fileInfo.getUrl();
                operationResult.postValue("图片上传成功");
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
        compareRepository.getAlgorithmOptions(new CompareRepository.OptionsCallback() {
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

    public void predict(long algorithmId, String params) {
        if (originalImageUrl == null) {
            error.setValue("请先上传图片");
            return;
        }
        PredParam param = new PredParam();
        param.setAlgorithmId(algorithmId);
        param.setImageUrl(originalImageUrl);
        param.setParams(params);
        loading.setValue(true);
        compareRepository.getPrediction(param, new CompareRepository.PredictionCallback() {
            @Override
            public void onSuccess(PredResult result) {
                predictionResult.postValue(result);
                operationResult.postValue("去雾处理完成");
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void predictMultiple(List<Long> algorithmIds, String params) {
        if (originalImageUrl == null) {
            error.setValue("请先上传图片");
            return;
        }
        if (algorithmIds == null || algorithmIds.isEmpty()) {
            error.setValue("请至少选择一个算法");
            return;
        }
        loading.setValue(true);
        Map<String, PredResult> results = new HashMap<>();
        final int[] pending = {algorithmIds.size()};
        for (Long algorithmId : algorithmIds) {
            PredParam param = new PredParam();
            param.setAlgorithmId(algorithmId);
            param.setImageUrl(originalImageUrl);
            param.setParams(params);
            compareRepository.getPrediction(param, new CompareRepository.PredictionCallback() {
                @Override
                public void onSuccess(PredResult result) {
                    synchronized (results) {
                        results.put(String.valueOf(algorithmId), result);
                    }
                    if (--pending[0] == 0) {
                        multiPredictionResults.postValue(results);
                        operationResult.postValue("多算法处理完成");
                        loading.postValue(false);
                    }
                }

                @Override
                public void onError(String errorMessage) {
                    if (--pending[0] == 0) {
                        multiPredictionResults.postValue(results);
                        loading.postValue(false);
                    }
                    error.postValue(errorMessage);
                }
            });
        }
    }

    public void evaluate(long algorithmId, String predUrl, String gtUrl) {
        EvalParam param = new EvalParam();
        param.setAlgorithmId(algorithmId);
        param.setPredUrl(predUrl);
        param.setGtUrl(gtUrl);
        loading.setValue(true);
        compareRepository.getEvaluation(param, new CompareRepository.EvaluationCallback() {
            @Override
            public void onSuccess(EvalResult result) {
                evaluationResult.postValue(result);
                operationResult.postValue("评估完成");
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void loadPredictionLogs() {
        loading.setValue(true);
        compareRepository.listPredictionLogs(1, 20, new CompareRepository.PredictionLogListCallback() {
            @Override
            public void onSuccess(List<PredictionLogVO> logs) {
                predictionLogs.postValue(logs != null ? logs : new ArrayList<>());
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
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

    public LiveData<Map<String, PredResult>> getMultiPredictionResults() {
        return multiPredictionResults;
    }

    public LiveData<EvalResult> getEvaluationResult() {
        return evaluationResult;
    }

    public LiveData<List<PredictionLogVO>> getPredictionLogs() {
        return predictionLogs;
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

    public String getOriginalImageUrl() {
        return originalImageUrl;
    }

    public void clearError() {
        error.setValue(null);
    }

    public void clearOperationResult() {
        operationResult.setValue(null);
    }
}
