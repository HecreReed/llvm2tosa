// Corresponding TOSA IR conversion - based on real LLVM IR structures

module {
  // 1. Simple scalar -> 1D tensor
  func.func @simple_add(%arg0: tensor<1xi32>, %arg1: tensor<1xi32>) -> tensor<1xi32> {
    %result = tosa.add %arg0, %arg1 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    return %result : tensor<1xi32>
  }

  // 2. Array operations -> tensor operations
  func.func @array_operations() {
    // Create initialized tensor (corresponds to LLVM alloca + store)
    %arr = tosa.const {value = dense<[1, 2, 0, 0]> : tensor<4xi32>} : () -> tensor<4xi32>
    
    // Extract first two elements (corresponds to LLVM load operations)
    %val0 = tosa.slice %arr {start = array<i64: 0>, size = array<i64: 1>} : (tensor<4xi32>) -> tensor<1xi32>
    %val1 = tosa.slice %arr {start = array<i64: 1>, size = array<i64: 1>} : (tensor<4xi32>) -> tensor<1xi32>
    
    // Addition operation
    %sum = tosa.add %val0, %val1 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    
    // Reconstruct tensor to update 3rd element (corresponds to LLVM store)
    %prefix = tosa.slice %arr {start = array<i64: 0>, size = array<i64: 2>} : (tensor<4xi32>) -> tensor<2xi32>
    %suffix = tosa.slice %arr {start = array<i64: 3>, size = array<i64: 1>} : (tensor<4xi32>) -> tensor<1xi32>
    %updated_arr = tosa.concat %prefix, %sum, %suffix {axis = 0 : i32} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    
    return
  }

  // 3. Vector operations -> direct tensor addition
  func.func @vector_add(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
    %result = tosa.add %arg0, %arg1 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
    return %result : tensor<4xi32>
  }

  // 4. Floating-point vector multiplication
  func.func @vector_multiply(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %shift = tosa.const {value = dense<0> : tensor<i8>} : () -> tensor<i8>
    %result = tosa.mul %arg0, %arg1, %shift : (tensor<4xf32>, tensor<4xf32>, tensor<i8>) -> tensor<4xf32>
    return %result : tensor<4xf32>
  }

  // 5. Matrix access -> 2D tensor operations
  func.func @matrix_access() -> tensor<1xi32> {
    // 2x3 matrix initialization
    %matrix = tosa.const {value = dense<[[0, 0, 0], [0, 0, 42]]> : tensor<2x3xi32>} : () -> tensor<2x3xi32>
    
    // Extract matrix[1][2] - last element
    %result = tosa.slice %matrix {start = array<i64: 1, 2>, size = array<i64: 1, 1>} 
              : (tensor<2x3xi32>) -> tensor<1x1xi32>
    
    // Reshape to 1D tensor
    %reshaped = tosa.reshape %result {new_shape = array<i64: 1>} : (tensor<1x1xi32>) -> tensor<1xi32>
    return %reshaped : tensor<1xi32>
  }

  // 6. Struct array -> composite tensor operations
  func.func @struct_with_array() {
    // Model struct: {array: [4xi32], scalar: i32}
    %array_part = tosa.const {value = dense<[0, 100, 0, 0]> : tensor<4xi32>} : () -> tensor<4xi32>
    %scalar_part = tosa.const {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    
    // In TOSA, we handle struct parts separately
    // Real implementation needs more complex memory layout handling
    return
  }

  // 7. Global array access -> constant tensor slice
  func.func @access_global_array(%arg0: tensor<1xi32>) -> tensor<1xi32> {
    %global_array = tosa.const {value = dense<[1, 2, 3, 4, 5]> : tensor<5xi32>} : () -> tensor<5xi32>
    
    // Dynamic indexing is complex in TOSA, this shows concept
    // Real implementation needs gather operations or conditional logic
    %result = tosa.slice %global_array {start = array<i64: 0>, size = array<i64: 1>} 
              : (tensor<5xi32>) -> tensor<1xi32>
    return %result : tensor<1xi32>
  }

  // 8. Vector element extraction -> tensor slice
  func.func @vector_extract(%arg0: tensor<4xi32>) -> tensor<1xi32> {
    %elem = tosa.slice %arg0 {start = array<i64: 2>, size = array<i64: 1>} 
            : (tensor<4xi32>) -> tensor<1xi32>
    return %elem : tensor<1xi32>
  }

  // 9. Vector element insertion -> tensor reconstruction
  func.func @vector_insert(%arg0: tensor<4xi32>, %arg1: tensor<1xi32>) -> tensor<4xi32> {
    %prefix = tosa.slice %arg0 {start = array<i64: 0>, size = array<i64: 1>} 
              : (tensor<4xi32>) -> tensor<1xi32>
    %suffix = tosa.slice %arg0 {start = array<i64: 2>, size = array<i64: 2>} 
              : (tensor<4xi32>) -> tensor<2xi32>
    
    %result = tosa.concat %prefix, %arg1, %suffix {axis = 0 : i32} 
              : (tensor<1xi32>, tensor<1xi32>, tensor<2xi32>) -> tensor<4xi32>
    return %result : tensor<4xi32>
  }

  // 10. Vector shuffle -> tensor rearrangement
  func.func @vector_shuffle(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
    // Extract required elements: a[0], a[2], b[0], b[2]
    %a0 = tosa.slice %arg0 {start = array<i64: 0>, size = array<i64: 1>} : (tensor<4xi32>) -> tensor<1xi32>
    %a2 = tosa.slice %arg0 {start = array<i64: 2>, size = array<i64: 1>} : (tensor<4xi32>) -> tensor<1xi32>
    %b0 = tosa.slice %arg1 {start = array<i64: 0>, size = array<i64: 1>} : (tensor<4xi32>) -> tensor<1xi32>
    %b2 = tosa.slice %arg1 {start = array<i64: 2>, size = array<i64: 1>} : (tensor<4xi32>) -> tensor<1xi32>
    
    %result = tosa.concat %a0, %a2, %b0, %b2 {axis = 0 : i32} 
              : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    return %result : tensor<4xi32>
  }
}