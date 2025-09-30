; 真实的LLVM IR示例 - 展示实际的数组和向量操作
; 这些是LLVM IR中确实存在的结构

; 1. 简单标量计算
define i32 @simple_add(i32 %a, i32 %b) {
entry:
  %result = add i32 %a, %b
  ret i32 %result
}

; 2. 数组操作 - LLVM IR中的真实数组处理
define void @array_operations() {
entry:
  ; 分配一个包含4个i32的数组
  %arr = alloca [4 x i32], align 16
  
  ; 获取数组元素的指针并存储值
  %ptr0 = getelementptr inbounds [4 x i32], [4 x i32]* %arr, i64 0, i64 0
  store i32 1, i32* %ptr0, align 4
  
  %ptr1 = getelementptr inbounds [4 x i32], [4 x i32]* %arr, i64 0, i64 1
  store i32 2, i32* %ptr1, align 4
  
  ; 加载并操作数组元素
  %val0 = load i32, i32* %ptr0, align 4
  %val1 = load i32, i32* %ptr1, align 4
  %sum = add i32 %val0, %val1
  
  ; 存储结果回数组
  %ptr2 = getelementptr inbounds [4 x i32], [4 x i32]* %arr, i64 0, i64 2
  store i32 %sum, i32* %ptr2, align 4
  
  ret void
}

; 3. 向量操作 - LLVM IR中的SIMD向量
define <4 x i32> @vector_add(<4 x i32> %a, <4 x i32> %b) {
entry:
  %result = add <4 x i32> %a, %b
  ret <4 x i32> %result
}

; 4. 浮点向量操作
define <4 x float> @vector_multiply(<4 x float> %a, <4 x float> %b) {
entry:
  %result = fmul <4 x float> %a, %b
  ret <4 x float> %result
}

; 5. 矩阵风格的嵌套数组
define i32 @matrix_access() {
entry:
  ; 2x3矩阵作为嵌套数组
  %matrix = alloca [2 x [3 x i32]], align 16
  
  ; 访问matrix[1][2]
  %ptr = getelementptr inbounds [2 x [3 x i32]], [2 x [3 x i32]]* %matrix, i64 0, i64 1, i64 2
  store i32 42, i32* %ptr, align 4
  
  %val = load i32, i32* %ptr, align 4
  ret i32 %val
}

; 6. 结构体包含数组
%struct.ArrayContainer = type { [4 x i32], i32 }

define void @struct_with_array() {
entry:
  %container = alloca %struct.ArrayContainer, align 16
  
  ; 访问结构体中的数组
  %arr_ptr = getelementptr inbounds %struct.ArrayContainer, %struct.ArrayContainer* %container, i32 0, i32 0
  %elem_ptr = getelementptr inbounds [4 x i32], [4 x i32]* %arr_ptr, i64 0, i64 1
  store i32 100, i32* %elem_ptr, align 4
  
  ret void
}

; 7. 全局数组
@global_array = global [5 x i32] [i32 1, i32 2, i32 3, i32 4, i32 5], align 16

define i32 @access_global_array(i32 %index) {
entry:
  %ptr = getelementptr inbounds [5 x i32], [5 x i32]* @global_array, i64 0, i32 %index
  %val = load i32, i32* %ptr, align 4
  ret i32 %val
}

; 8. 向量元素提取和插入
define i32 @vector_extract(<4 x i32> %vec) {
entry:
  %elem = extractelement <4 x i32> %vec, i32 2
  ret i32 %elem
}

define <4 x i32> @vector_insert(<4 x i32> %vec, i32 %val) {
entry:
  %result = insertelement <4 x i32> %vec, i32 %val, i32 1
  ret <4 x i32> %result
}

; 9. 向量洗牌操作
define <4 x i32> @vector_shuffle(<4 x i32> %a, <4 x i32> %b) {
entry:
  %result = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  ret <4 x i32> %result
}