; Example LLVM IR with array operations for testing TOSA conversion
; This demonstrates basic tensor-like operations in LLVM IR

define i32 @vector_add_example() {
entry:
  ; Create two arrays (will be converted to tensors in TOSA)
  %arr1 = alloca [4 x i32], align 16
  %arr2 = alloca [4 x i32], align 16
  %result = alloca [4 x i32], align 16
  
  ; Initialize first array: [1, 2, 3, 4]
  %ptr1_0 = getelementptr inbounds [4 x i32], [4 x i32]* %arr1, i64 0, i64 0
  store i32 1, i32* %ptr1_0, align 4
  %ptr1_1 = getelementptr inbounds [4 x i32], [4 x i32]* %arr1, i64 0, i64 1
  store i32 2, i32* %ptr1_1, align 4
  %ptr1_2 = getelementptr inbounds [4 x i32], [4 x i32]* %arr1, i64 0, i64 2
  store i32 3, i32* %ptr1_2, align 4
  %ptr1_3 = getelementptr inbounds [4 x i32], [4 x i32]* %arr1, i64 0, i64 3
  store i32 4, i32* %ptr1_3, align 4
  
  ; Initialize second array: [5, 6, 7, 8]
  %ptr2_0 = getelementptr inbounds [4 x i32], [4 x i32]* %arr2, i64 0, i64 0
  store i32 5, i32* %ptr2_0, align 4
  %ptr2_1 = getelementptr inbounds [4 x i32], [4 x i32]* %arr2, i64 0, i64 1
  store i32 6, i32* %ptr2_1, align 4
  %ptr2_2 = getelementptr inbounds [4 x i32], [4 x i32]* %arr2, i64 0, i64 2
  store i32 7, i32* %ptr2_2, align 4
  %ptr2_3 = getelementptr inbounds [4 x i32], [4 x i32]* %arr2, i64 0, i64 3
  store i32 8, i32* %ptr2_3, align 4
  
  ; Perform element-wise addition: result[i] = arr1[i] + arr2[i]
  %val1_0 = load i32, i32* %ptr1_0, align 4
  %val2_0 = load i32, i32* %ptr2_0, align 4
  %sum_0 = add i32 %val1_0, %val2_0
  %res_ptr_0 = getelementptr inbounds [4 x i32], [4 x i32]* %result, i64 0, i64 0
  store i32 %sum_0, i32* %res_ptr_0, align 4
  
  %val1_1 = load i32, i32* %ptr1_1, align 4
  %val2_1 = load i32, i32* %ptr2_1, align 4
  %sum_1 = add i32 %val1_1, %val2_1
  %res_ptr_1 = getelementptr inbounds [4 x i32], [4 x i32]* %result, i64 0, i64 1
  store i32 %sum_1, i32* %res_ptr_1, align 4
  
  %val1_2 = load i32, i32* %ptr1_2, align 4
  %val2_2 = load i32, i32* %ptr2_2, align 4
  %sum_2 = add i32 %val1_2, %val2_2
  %res_ptr_2 = getelementptr inbounds [4 x i32], [4 x i32]* %result, i64 0, i64 2
  store i32 %sum_2, i32* %res_ptr_2, align 4
  
  %val1_3 = load i32, i32* %ptr1_3, align 4
  %val2_3 = load i32, i32* %ptr2_3, align 4
  %sum_3 = add i32 %val1_3, %val2_3
  %res_ptr_3 = getelementptr inbounds [4 x i32], [4 x i32]* %result, i64 0, i64 3
  store i32 %sum_3, i32* %res_ptr_3, align 4
  
  ; Return first element of result for verification
  %final_result = load i32, i32* %res_ptr_0, align 4
  ret i32 %final_result
}

define float @float_vector_ops() {
entry:
  ; Float array operations
  %farr1 = alloca [3 x float], align 16
  %farr2 = alloca [3 x float], align 16
  
  ; Initialize float arrays
  %fptr1_0 = getelementptr inbounds [3 x float], [3 x float]* %farr1, i64 0, i64 0
  store float 1.5, float* %fptr1_0, align 4
  %fptr1_1 = getelementptr inbounds [3 x float], [3 x float]* %farr1, i64 0, i64 1
  store float 2.5, float* %fptr1_1, align 4
  %fptr1_2 = getelementptr inbounds [3 x float], [3 x float]* %farr1, i64 0, i64 2
  store float 3.5, float* %fptr1_2, align 4
  
  %fptr2_0 = getelementptr inbounds [3 x float], [3 x float]* %farr2, i64 0, i64 0
  store float 0.5, float* %fptr2_0, align 4
  %fptr2_1 = getelementptr inbounds [3 x float], [3 x float]* %farr2, i64 0, i64 1
  store float 1.0, float* %fptr2_1, align 4
  %fptr2_2 = getelementptr inbounds [3 x float], [3 x float]* %farr2, i64 0, i64 2
  store float 1.5, float* %fptr2_2, align 4
  
  ; Perform multiplication
  %fval1_0 = load float, float* %fptr1_0, align 4
  %fval2_0 = load float, float* %fptr2_0, align 4
  %fmul_0 = fmul float %fval1_0, %fval2_0
  
  ; Perform subtraction
  %fval1_1 = load float, float* %fptr1_1, align 4
  %fval2_1 = load float, float* %fptr2_1, align 4
  %fsub_1 = fsub float %fval1_1, %fval2_1
  
  ; Final operation: multiply results
  %final_float = fmul float %fmul_0, %fsub_1
  ret float %final_float
}

; Simple scalar arithmetic that should convert to tensor operations
define i32 @scalar_ops(i32 %a, i32 %b, i32 %c) {
entry:
  %add_result = add i32 %a, %b
  %sub_result = sub i32 %add_result, %c
  %mul_result = mul i32 %sub_result, 2
  ret i32 %mul_result
}

; Matrix-like operations using nested arrays
define i32 @matrix_example() {
entry:
  ; 2x2 matrix represented as nested arrays
  %matrix = alloca [2 x [2 x i32]], align 16
  
  ; Initialize matrix [[1, 2], [3, 4]]
  %m00 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %matrix, i64 0, i64 0, i64 0
  store i32 1, i32* %m00, align 4
  %m01 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %matrix, i64 0, i64 0, i64 1
  store i32 2, i32* %m01, align 4
  %m10 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %matrix, i64 0, i64 1, i64 0
  store i32 3, i32* %m10, align 4
  %m11 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %matrix, i64 0, i64 1, i64 1
  store i32 4, i32* %m11, align 4
  
  ; Load and sum diagonal elements
  %val00 = load i32, i32* %m00, align 4
  %val11 = load i32, i32* %m11, align 4
  %diagonal_sum = add i32 %val00, %val11
  
  ret i32 %diagonal_sum
}