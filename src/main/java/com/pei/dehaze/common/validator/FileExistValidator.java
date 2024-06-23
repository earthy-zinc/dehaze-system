package com.pei.dehaze.common.validator;

import jakarta.validation.ConstraintValidator;
import jakarta.validation.ConstraintValidatorContext;

import java.io.File;

public class FileExistValidator implements ConstraintValidator<FileExists, String> {

    @Override
    public boolean isValid(String value, ConstraintValidatorContext context) {
        File file = new File(value);
        return file.exists() && file.isFile(); // 路径存在且为目录
    }
}
