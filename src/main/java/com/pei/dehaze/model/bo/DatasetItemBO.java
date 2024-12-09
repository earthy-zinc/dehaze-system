package com.pei.dehaze.model.bo;

import lombok.Data;
import lombok.EqualsAndHashCode;

@EqualsAndHashCode(callSuper = true)
@Data
public class DatasetItemBO extends FileBO{
    private String type;
    private String description;
}
