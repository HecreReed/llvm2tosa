// Expected TOSA IR output after conversion
// This shows what the converter should produce for the example LLVM IR

module {
  func.func @vector_add_example() -> tensor<1xi32> {
    // Constants representing the initial arrays
    %arr1 = tosa.const {value = dense<[1, 2, 3, 4]> : tensor<4xi32>} : () -> tensor<4xi32>
    %arr2 = tosa.const {value = dense<[5, 6, 7, 8]> : tensor<4xi32>} : () -> tensor<4xi32>
    
    // Element-wise addition using TOSA add operation
    %result = tosa.add %arr1, %arr2 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
    
    // Extract first element (slice operation)
    %first_element = tosa.slice %result {start = array<i64: 0>, size = array<i64: 1>} 
                     : (tensor<4xi32>) -> tensor<1xi32>
    
    return %first_element : tensor<1xi32>
  }

  func.func @float_vector_ops() -> tensor<1xf32> {
    // Float tensor constants
    %farr1 = tosa.const {value = dense<[1.5, 2.5, 3.5]> : tensor<3xf32>} : () -> tensor<3xf32>
    %farr2 = tosa.const {value = dense<[0.5, 1.0, 1.5]> : tensor<3xf32>} : () -> tensor<3xf32>
    
    // Element-wise multiplication
    %zero_shift = tosa.const {value = dense<0> : tensor<i8>} : () -> tensor<i8>
    %mul_result = tosa.mul %farr1, %farr2, %zero_shift 
                  : (tensor<3xf32>, tensor<3xf32>, tensor<i8>) -> tensor<3xf32>
    
    // Extract and process elements
    %elem0 = tosa.slice %mul_result {start = array<i64: 0>, size = array<i64: 1>} 
             : (tensor<3xf32>) -> tensor<1xf32>
    %elem1_farr1 = tosa.slice %farr1 {start = array<i64: 1>, size = array<i64: 1>} 
                   : (tensor<3xf32>) -> tensor<1xf32>
    %elem1_farr2 = tosa.slice %farr2 {start = array<i64: 1>, size = array<i64: 1>} 
                   : (tensor<3xf32>) -> tensor<1xf32>
    
    // Subtraction
    %sub_result = tosa.sub %elem1_farr1, %elem1_farr2 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    
    // Final multiplication
    %final_result = tosa.mul %elem0, %sub_result, %zero_shift 
                    : (tensor<1xf32>, tensor<1xf32>, tensor<i8>) -> tensor<1xf32>
    
    return %final_result : tensor<1xf32>
  }

  func.func @scalar_ops(%arg0: tensor<1xi32>, %arg1: tensor<1xi32>, %arg2: tensor<1xi32>) -> tensor<1xi32> {
    // Scalar operations converted to tensor operations
    %add_result = tosa.add %arg0, %arg1 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %sub_result = tosa.sub %add_result, %arg2 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    
    %two = tosa.const {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
    %zero_shift = tosa.const {value = dense<0> : tensor<i8>} : () -> tensor<i8>
    %mul_result = tosa.mul %sub_result, %two, %zero_shift 
                  : (tensor<1xi32>, tensor<1xi32>, tensor<i8>) -> tensor<1xi32>
    
    return %mul_result : tensor<1xi32>
  }

  func.func @matrix_example() -> tensor<1xi32> {
    // 2x2 matrix as a tensor
    %matrix = tosa.const {value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
    
    // Extract diagonal elements
    %elem00 = tosa.slice %matrix {start = array<i64: 0, 0>, size = array<i64: 1, 1>} 
              : (tensor<2x2xi32>) -> tensor<1x1xi32>
    %elem11 = tosa.slice %matrix {start = array<i64: 1, 1>, size = array<i64: 1, 1>} 
              : (tensor<2x2xi32>) -> tensor<1x1xi32>
    
    // Reshape to 1D for addition
    %elem00_1d = tosa.reshape %elem00 {new_shape = array<i64: 1>} : (tensor<1x1xi32>) -> tensor<1xi32>
    %elem11_1d = tosa.reshape %elem11 {new_shape = array<i64: 1>} : (tensor<1x1xi32>) -> tensor<1xi32>
    
    // Add diagonal elements
    %diagonal_sum = tosa.add %elem00_1d, %elem11_1d : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    
    return %diagonal_sum : tensor<1xi32>
  }
}