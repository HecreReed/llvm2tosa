// 对应的TOSA IR转换 - 基于真实的LLVM IR结构

module {
  // 1. 简单标量 -> 1D张量
  func.func @simple_add(%arg0: tensor<1xi32>, %arg1: tensor<1xi32>) -> tensor<1xi32> {
    %result = tosa.add %arg0, %arg1 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    return %result : tensor<1xi32>
  }

  // 2. 数组操作 -> 张量操作
  func.func @array_operations() {
    // 创建初始化的张量 (对应LLVM的alloca + store)
    %arr = tosa.const {value = dense<[1, 2, 0, 0]> : tensor<4xi32>} : () -> tensor<4xi32>
    
    // 提取前两个元素 (对应LLVM的load操作)
    %val0 = tosa.slice %arr {start = array<i64: 0>, size = array<i64: 1>} : (tensor<4xi32>) -> tensor<1xi32>
    %val1 = tosa.slice %arr {start = array<i64: 1>, size = array<i64: 1>} : (tensor<4xi32>) -> tensor<1xi32>
    
    // 加法操作
    %sum = tosa.add %val0, %val1 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    
    // 重构张量以更新第3个元素 (对应LLVM的store操作)
    %prefix = tosa.slice %arr {start = array<i64: 0>, size = array<i64: 2>} : (tensor<4xi32>) -> tensor<2xi32>
    %suffix = tosa.slice %arr {start = array<i64: 3>, size = array<i64: 1>} : (tensor<4xi32>) -> tensor<1xi32>
    %updated_arr = tosa.concat %prefix, %sum, %suffix {axis = 0 : i32} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    
    return
  }

  // 3. 向量操作 -> 直接张量加法
  func.func @vector_add(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
    %result = tosa.add %arg0, %arg1 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
    return %result : tensor<4xi32>
  }

  // 4. 浮点向量乘法
  func.func @vector_multiply(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %shift = tosa.const {value = dense<0> : tensor<i8>} : () -> tensor<i8>
    %result = tosa.mul %arg0, %arg1, %shift : (tensor<4xf32>, tensor<4xf32>, tensor<i8>) -> tensor<4xf32>
    return %result : tensor<4xf32>
  }

  // 5. 矩阵访问 -> 2D张量操作
  func.func @matrix_access() -> tensor<1xi32> {
    // 2x3矩阵初始化
    %matrix = tosa.const {value = dense<[[0, 0, 0], [0, 0, 42]]> : tensor<2x3xi32>} : () -> tensor<2x3xi32>
    
    // 提取matrix[1][2] - 最后一个元素
    %result = tosa.slice %matrix {start = array<i64: 1, 2>, size = array<i64: 1, 1>} 
              : (tensor<2x3xi32>) -> tensor<1x1xi32>
    
    // 重塑为1D张量
    %reshaped = tosa.reshape %result {new_shape = array<i64: 1>} : (tensor<1x1xi32>) -> tensor<1xi32>
    return %reshaped : tensor<1xi32>
  }

  // 6. 结构体数组 -> 复合张量操作
  func.func @struct_with_array() {
    // 模拟结构体：{array: [4xi32], scalar: i32}
    %array_part = tosa.const {value = dense<[0, 100, 0, 0]> : tensor<4xi32>} : () -> tensor<4xi32>
    %scalar_part = tosa.const {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    
    // 在TOSA中，我们分别处理结构体的各个部分
    // 真实实现需要更复杂的内存布局处理
    return
  }

  // 7. 全局数组访问 -> 常量张量切片
  func.func @access_global_array(%arg0: tensor<1xi32>) -> tensor<1xi32> {
    %global_array = tosa.const {value = dense<[1, 2, 3, 4, 5]> : tensor<5xi32>} : () -> tensor<5xi32>
    
    // 动态索引在TOSA中较复杂，这里展示概念
    // 实际需要使用gather操作或条件逻辑
    %result = tosa.slice %global_array {start = array<i64: 0>, size = array<i64: 1>} 
              : (tensor<5xi32>) -> tensor<1xi32>
    return %result : tensor<1xi32>
  }

  // 8. 向量元素提取 -> 张量切片
  func.func @vector_extract(%arg0: tensor<4xi32>) -> tensor<1xi32> {
    %elem = tosa.slice %arg0 {start = array<i64: 2>, size = array<i64: 1>} 
            : (tensor<4xi32>) -> tensor<1xi32>
    return %elem : tensor<1xi32>
  }

  // 9. 向量元素插入 -> 张量重构
  func.func @vector_insert(%arg0: tensor<4xi32>, %arg1: tensor<1xi32>) -> tensor<4xi32> {
    %prefix = tosa.slice %arg0 {start = array<i64: 0>, size = array<i64: 1>} 
              : (tensor<4xi32>) -> tensor<1xi32>
    %suffix = tosa.slice %arg0 {start = array<i64: 2>, size = array<i64: 2>} 
              : (tensor<4xi32>) -> tensor<2xi32>
    
    %result = tosa.concat %prefix, %arg1, %suffix {axis = 0 : i32} 
              : (tensor<1xi32>, tensor<1xi32>, tensor<2xi32>) -> tensor<4xi32>
    return %result : tensor<4xi32>
  }

  // 10. 向量洗牌 -> 张量重排
  func.func @vector_shuffle(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
    // 提取所需元素: a[0], a[2], b[0], b[2]
    %a0 = tosa.slice %arg0 {start = array<i64: 0>, size = array<i64: 1>} : (tensor<4xi32>) -> tensor<1xi32>
    %a2 = tosa.slice %arg0 {start = array<i64: 2>, size = array<i64: 1>} : (tensor<4xi32>) -> tensor<1xi32>
    %b0 = tosa.slice %arg1 {start = array<i64: 0>, size = array<i64: 1>} : (tensor<4xi32>) -> tensor<1xi32>
    %b2 = tosa.slice %arg1 {start = array<i64: 2>, size = array<i64: 1>} : (tensor<4xi32>) -> tensor<1xi32>
    
    %result = tosa.concat %a0, %a2, %b0, %b2 {axis = 0 : i32} 
              : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    return %result : tensor<4xi32>
  }
}