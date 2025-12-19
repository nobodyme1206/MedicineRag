package com.medical.rag.dto;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Data;

@Data
public class AgentRequest {
    @NotBlank(message = "查询不能为空")
    @Size(max = 2000, message = "查询长度不能超过2000字符")
    private String query;
    
    @Min(1) @Max(10)
    private int maxSteps = 5;
    
    private boolean verbose = true;
}
