package com.pei.dehaze.ui.task.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.pei.dehaze.repository.TaskRepository;
import com.pei.dehaze.sdk.model.task.TaskCreateForm;
import com.pei.dehaze.sdk.model.task.TaskQuery;
import com.pei.dehaze.sdk.model.task.TaskVO;

import java.util.ArrayList;
import java.util.List;

/**
 * 任务管理 ViewModel
 */
public class TaskViewModel extends ViewModel {

    private final TaskRepository taskRepository;

    private final MutableLiveData<List<TaskVO>> taskList = new MutableLiveData<>();
    private final MutableLiveData<Boolean> loading = new MutableLiveData<>();
    private final MutableLiveData<String> error = new MutableLiveData<>();
    private final MutableLiveData<String> operationResult = new MutableLiveData<>();
    private final MutableLiveData<TaskVO> taskDetail = new MutableLiveData<>();
    private final MutableLiveData<TaskVO> createdTask = new MutableLiveData<>();

    private int pageNum = 1;
    private int pageSize = 10;
    private String statusFilter;
    private String typeFilter;
    private long total = 0;

    public TaskViewModel() {
        taskRepository = new TaskRepository();
    }

    /**
     * 加载任务列表（首页）
     */
    public void loadTasks() {
        pageNum = 1;
        fetchTasks();
    }

    /**
     * 按状态筛选
     */
    public void filterByStatus(String status) {
        this.statusFilter = status;
        pageNum = 1;
        fetchTasks();
    }

    /**
     * 按类型筛选
     */
    public void filterByType(String taskType) {
        this.typeFilter = taskType;
        pageNum = 1;
        fetchTasks();
    }

    /**
     * 加载下一页
     */
    public void loadMore() {
        if (taskList.getValue() == null || taskList.getValue().size() >= total) {
            return;
        }
        pageNum++;
        fetchTasks();
    }

    private void fetchTasks() {
        loading.setValue(true);
        TaskQuery query = new TaskQuery();
        query.setPageNum(pageNum);
        query.setPageSize(pageSize);
        query.setStatus(statusFilter);
        query.setTaskType(typeFilter);
        taskRepository.getTasks(query, new TaskRepository.TaskListCallback() {
            @Override
            public void onSuccess(List<TaskVO> tasks, long total) {
                TaskViewModel.this.total = total;
                if (pageNum == 1) {
                    taskList.postValue(tasks);
                } else {
                    List<TaskVO> current = taskList.getValue();
                    if (current == null) {
                        current = new ArrayList<>();
                    }
                    current.addAll(tasks);
                    taskList.postValue(current);
                }
                loading.postValue(false);
            }

            @Override
            public void onError(String code, String message) {
                error.postValue("[" + code + "] " + message);
                loading.postValue(false);
            }
        });
    }

    /**
     * 创建任务
     */
    public void createTask(TaskCreateForm form) {
        loading.setValue(true);
        taskRepository.createTask(form, new TaskRepository.TaskCallback() {
            @Override
            public void onSuccess(TaskVO task) {
                createdTask.postValue(task);
                operationResult.postValue("任务创建成功: " + task.getTaskId());
                loading.postValue(false);
                loadTasks();
            }

            @Override
            public void onError(String code, String message) {
                error.postValue("创建失败: [" + code + "] " + message);
                loading.postValue(false);
            }
        });
    }

    /**
     * 取消任务
     */
    public void cancelTask(String taskId) {
        loading.setValue(true);
        taskRepository.cancelTask(taskId, new TaskRepository.ActionCallback() {
            @Override
            public void onSuccess() {
                operationResult.postValue("任务已取消");
                loading.postValue(false);
                loadTasks();
            }

            @Override
            public void onError(String code, String message) {
                error.postValue("取消失败: [" + code + "] " + message);
                loading.postValue(false);
            }
        });
    }

    /**
     * 下载任务结果
     */
    public void downloadTaskFile(String taskId) {
        loading.setValue(true);
        taskRepository.downloadTaskFile(taskId, new TaskRepository.ActionCallback() {
            @Override
            public void onSuccess() {
                operationResult.postValue("下载成功，已保存到下载目录");
                loading.postValue(false);
            }

            @Override
            public void onError(String code, String message) {
                error.postValue("下载失败: [" + code + "] " + message);
                loading.postValue(false);
            }
        });
    }

    /**
     * 删除任务
     */
    public void deleteTask(long id) {
        loading.setValue(true);
        taskRepository.deleteTask(id, new TaskRepository.ActionCallback() {
            @Override
            public void onSuccess() {
                operationResult.postValue("删除成功");
                loading.postValue(false);
                loadTasks();
            }

            @Override
            public void onError(String code, String message) {
                error.postValue("删除失败: [" + code + "] " + message);
                loading.postValue(false);
            }
        });
    }

    /**
     * 查看任务详情
     */
    public void getTaskDetail(String taskId) {
        loading.setValue(true);
        taskRepository.getTask(taskId, new TaskRepository.TaskCallback() {
            @Override
            public void onSuccess(TaskVO task) {
                taskDetail.postValue(task);
                loading.postValue(false);
            }

            @Override
            public void onError(String code, String message) {
                error.postValue("查询详情失败: [" + code + "] " + message);
                loading.postValue(false);
            }
        });
    }

    public LiveData<List<TaskVO>> getTaskList() {
        return taskList;
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

    public LiveData<TaskVO> getTaskDetail() {
        return taskDetail;
    }

    public LiveData<TaskVO> getCreatedTask() {
        return createdTask;
    }

    public long getTotal() {
        return total;
    }
}
