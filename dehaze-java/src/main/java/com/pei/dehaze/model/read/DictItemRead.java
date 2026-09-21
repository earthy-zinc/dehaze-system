package com.pei.dehaze.model.read;

import lombok.Data;

/**
 * 字典项（name → value）读取行
 *
 * @author dehaze
 */
@Data
public class DictItemRead {

    private String name;

    private String value;
}
