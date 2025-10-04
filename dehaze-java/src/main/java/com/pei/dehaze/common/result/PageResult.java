package com.pei.dehaze.common.result;

import com.baomidou.mybatisplus.core.metadata.IPage;
import lombok.Data;
import org.springframework.data.domain.Page;

import java.io.Serializable;
import java.util.List;

/**
 * 分页响应结构体
 *
 * @author earthyzinc
 * @since 2022/2/18 23:29
 */
@Data
public class PageResult<T> implements Serializable {

    private String code;

    private Data<T> data;

    private String msg;

    public static <T> PageResult<T> success(Page<T> page) {
        PageResult<T> result = new PageResult<>();
        result.setCode(ResultCode.SUCCESS.getCode());

        Data<T> data = new Data<>();
        data.setList(page.getContent());
        data.setTotal(page.getTotalElements());

        result.setData(data);
        result.setMsg(ResultCode.SUCCESS.getMsg());
        return result;
    }

    public static <T> PageResult<T> success(IPage<T> page) {
        PageResult<T> result = new PageResult<>();
        result.setCode(ResultCode.SUCCESS.getCode());

        Data<T> data = new Data<>();
        data.setList(page.getRecords());
        data.setTotal(page.getTotal());

        result.setData(data);
        result.setMsg(ResultCode.SUCCESS.getMsg());
        return result;
    }

    @lombok.Data
    public static class Data<T> implements Serializable {

        private List<T> list;

        private long total;

    }

}
