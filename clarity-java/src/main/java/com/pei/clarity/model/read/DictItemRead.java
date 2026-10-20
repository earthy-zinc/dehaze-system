package com.pei.clarity.model.read;

import lombok.Data;

/**
 * 字典项（name → value）读取行
 *
 * @author clarity
 */
@Data
public class DictItemRead {

    private String name;

    private String value;
}
