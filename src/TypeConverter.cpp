#include "LLVMToTosaConverter.h"
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/Support/Casting.h>

namespace llvm2tosa {

TypeConverter::TypeConverter(mlir::MLIRContext& context) : context_(context) {}

mlir::Type TypeConverter::convertType(llvm::Type* llvmType) {
    // Check cache first
    auto it = typeCache_.find(llvmType);
    if (it != typeCache_.end()) {
        return it->second;
    }
    
    mlir::Type result;
    
    switch (llvmType->getTypeID()) {
        case llvm::Type::VoidTyID:
            result = mlir::NoneType::get(&context_);
            break;
            
        case llvm::Type::IntegerTyID:
            result = convertIntegerType(llvm::cast<llvm::IntegerType>(llvmType));
            break;
            
        case llvm::Type::HalfTyID:
        case llvm::Type::BFloatTyID:
        case llvm::Type::FloatTyID:
        case llvm::Type::DoubleTyID:
        case llvm::Type::X86_FP80TyID:
        case llvm::Type::FP128TyID:
        case llvm::Type::PPC_FP128TyID:
            result = convertFloatingPointType(llvmType);
            break;
            
        case llvm::Type::PointerTyID:
            result = convertPointerType(llvm::cast<llvm::PointerType>(llvmType));
            break;
            
        case llvm::Type::ArrayTyID:
            result = convertArrayType(llvm::cast<llvm::ArrayType>(llvmType));
            break;
            
        case llvm::Type::FixedVectorTyID:
        case llvm::Type::ScalableVectorTyID:
            result = convertVectorType(llvm::cast<llvm::VectorType>(llvmType));
            break;
            
        case llvm::Type::StructTyID:
            result = convertStructType(llvm::cast<llvm::StructType>(llvmType));
            break;
            
        case llvm::Type::FunctionTyID:
            result = convertFunctionType(llvm::cast<llvm::FunctionType>(llvmType));
            break;
            
        default:
            // Fallback: treat as opaque tensor
            result = mlir::RankedTensorType::get({1}, mlir::FloatType::getF32(&context_));
            break;
    }
    
    // Cache the result
    typeCache_[llvmType] = result;
    return result;
}

mlir::Type TypeConverter::convertIntegerType(llvm::IntegerType* intType) {
    unsigned bitWidth = intType->getBitWidth();
    
    // LLVM scalar integer -> TOSA 1D tensor
    mlir::Type elementType;
    
    switch (bitWidth) {
        case 1:
            // Boolean -> i1 tensor
            elementType = mlir::IntegerType::get(&context_, 1);
            break;
        case 8:
            elementType = mlir::IntegerType::get(&context_, 8);
            break;
        case 16:
            elementType = mlir::IntegerType::get(&context_, 16);
            break;
        case 32:
            elementType = mlir::IntegerType::get(&context_, 32);
            break;
        case 64:
            elementType = mlir::IntegerType::get(&context_, 64);
            break;
        default:
            // For unusual bit widths, round up to next power of 2
            if (bitWidth <= 8) elementType = mlir::IntegerType::get(&context_, 8);
            else if (bitWidth <= 16) elementType = mlir::IntegerType::get(&context_, 16);
            else if (bitWidth <= 32) elementType = mlir::IntegerType::get(&context_, 32);
            else elementType = mlir::IntegerType::get(&context_, 64);
            break;
    }
    
    return mlir::RankedTensorType::get({1}, elementType);
}

mlir::Type TypeConverter::convertFloatingPointType(llvm::Type* fpType) {
    mlir::Type elementType;
    
    switch (fpType->getTypeID()) {
        case llvm::Type::HalfTyID:
            elementType = mlir::FloatType::getF16(&context_);
            break;
        case llvm::Type::BFloatTyID:
            elementType = mlir::FloatType::getBF16(&context_);
            break;
        case llvm::Type::FloatTyID:
            elementType = mlir::FloatType::getF32(&context_);
            break;
        case llvm::Type::DoubleTyID:
            elementType = mlir::FloatType::getF64(&context_);
            break;
        default:
            // For extended precision types, use F64
            elementType = mlir::FloatType::getF64(&context_);
            break;
    }
    
    return mlir::RankedTensorType::get({1}, elementType);
}

mlir::Type TypeConverter::convertPointerType(llvm::PointerType* ptrType) {
    // Pointers in LLVM become references to tensors in TOSA
    // The pointed-to type determines the tensor element type
    llvm::Type* pointeeType = ptrType->getPointerElementType();
    
    // Convert the pointee type and create a tensor reference
    mlir::Type pointeeMLIRType = convertType(pointeeType);
    
    // If pointee is already a tensor, return it directly
    if (auto tensorType = pointeeMLIRType.dyn_cast<mlir::RankedTensorType>()) {
        return tensorType;
    }
    
    // Otherwise, create a tensor type from the pointee
    if (pointeeType->isIntegerTy()) {
        auto intType = llvm::cast<llvm::IntegerType>(pointeeType);
        auto elementType = mlir::IntegerType::get(&context_, intType->getBitWidth());
        return mlir::RankedTensorType::get({1}, elementType);
    } else if (pointeeType->isFloatingPointTy()) {
        auto elementType = convertFloatingPointType(pointeeType).cast<mlir::RankedTensorType>().getElementType();
        return mlir::RankedTensorType::get({1}, elementType);
    }
    
    // Default fallback
    return mlir::RankedTensorType::get({1}, mlir::FloatType::getF32(&context_));
}

mlir::Type TypeConverter::convertArrayType(llvm::ArrayType* arrayType) {
    // LLVM array -> TOSA multi-dimensional tensor
    std::vector<int64_t> shape = inferTensorShape(arrayType);
    mlir::Type elementType = getElementType(arrayType);
    
    return mlir::RankedTensorType::get(shape, elementType);
}

mlir::Type TypeConverter::convertVectorType(llvm::VectorType* vectorType) {
    // LLVM vector -> TOSA 1D or multi-dimensional tensor
    std::vector<int64_t> shape;
    
    if (auto fixedVectorType = llvm::dyn_cast<llvm::FixedVectorType>(vectorType)) {
        shape.push_back(fixedVectorType->getNumElements());
    } else if (auto scalableVectorType = llvm::dyn_cast<llvm::ScalableVectorType>(vectorType)) {
        // For scalable vectors, we need to use a dynamic shape or fixed estimate
        // For now, use a fixed size estimate
        shape.push_back(scalableVectorType->getMinNumElements() * 4); // Estimate
    }
    
    mlir::Type elementType = getElementType(vectorType->getElementType());
    return mlir::RankedTensorType::get(shape, elementType);
}

mlir::Type TypeConverter::convertStructType(llvm::StructType* structType) {
    // LLVM struct -> Multiple TOSA tensors or packed tensor
    // For now, convert to a single tensor with flattened layout
    
    size_t totalElements = 0;
    mlir::Type commonElementType = mlir::FloatType::getF32(&context_);
    
    // Calculate total size and determine common type
    for (unsigned i = 0; i < structType->getNumElements(); ++i) {
        llvm::Type* elemType = structType->getElementType(i);
        totalElements += getTensorElementCount(elemType);
        
        // Use the most common type (for simplicity, use f32)
        if (elemType->isFloatingPointTy()) {
            commonElementType = mlir::FloatType::getF32(&context_);
        } else if (elemType->isIntegerTy()) {
            commonElementType = mlir::IntegerType::get(&context_, 32);
        }
    }
    
    return mlir::RankedTensorType::get({static_cast<int64_t>(totalElements)}, commonElementType);
}

mlir::FunctionType TypeConverter::convertFunctionType(llvm::FunctionType* llvmFuncType) {
    // Convert parameter types
    llvm::SmallVector<mlir::Type> paramTypes;
    for (unsigned i = 0; i < llvmFuncType->getNumParams(); ++i) {
        paramTypes.push_back(convertType(llvmFuncType->getParamType(i)));
    }
    
    // Convert return type
    llvm::SmallVector<mlir::Type> resultTypes;
    llvm::Type* returnType = llvmFuncType->getReturnType();
    if (!returnType->isVoidTy()) {
        resultTypes.push_back(convertType(returnType));
    }
    
    return mlir::FunctionType::get(&context_, paramTypes, resultTypes);
}

std::vector<int64_t> TypeConverter::inferTensorShape(llvm::Type* llvmType) {
    std::vector<int64_t> shape;
    
    if (auto arrayType = llvm::dyn_cast<llvm::ArrayType>(llvmType)) {
        shape.push_back(arrayType->getNumElements());
        
        // Recursively handle nested arrays
        llvm::Type* elementType = arrayType->getElementType();
        if (elementType->isArrayTy()) {
            auto nestedShape = inferTensorShape(elementType);
            shape.insert(shape.end(), nestedShape.begin(), nestedShape.end());
        }
    } else if (auto vectorType = llvm::dyn_cast<llvm::FixedVectorType>(llvmType)) {
        shape.push_back(vectorType->getNumElements());
    } else {
        // Scalar types become 1D tensors
        shape.push_back(1);
    }
    
    return shape;
}

mlir::Type TypeConverter::getElementType(llvm::Type* llvmType) {
    // Extract the leaf element type from nested structures
    llvm::Type* currentType = llvmType;
    
    while (currentType->isArrayTy()) {
        currentType = llvm::cast<llvm::ArrayType>(currentType)->getElementType();
    }
    
    if (currentType->isVectorTy()) {
        currentType = llvm::cast<llvm::VectorType>(currentType)->getElementType();
    }
    
    // Convert the final element type
    if (currentType->isIntegerTy()) {
        auto intType = llvm::cast<llvm::IntegerType>(currentType);
        return mlir::IntegerType::get(&context_, intType->getBitWidth());
    } else if (currentType->isFloatingPointTy()) {
        switch (currentType->getTypeID()) {
            case llvm::Type::HalfTyID:
                return mlir::FloatType::getF16(&context_);
            case llvm::Type::BFloatTyID:
                return mlir::FloatType::getBF16(&context_);
            case llvm::Type::FloatTyID:
                return mlir::FloatType::getF32(&context_);
            case llvm::Type::DoubleTyID:
                return mlir::FloatType::getF64(&context_);
            default:
                return mlir::FloatType::getF32(&context_);
        }
    }
    
    // Default fallback
    return mlir::FloatType::getF32(&context_);
}

namespace utils {

bool isDirectlyConvertible(llvm::Type* llvmType) {
    return llvmType->isIntegerTy() || 
           llvmType->isFloatingPointTy() || 
           llvmType->isVectorTy() || 
           llvmType->isArrayTy();
}

size_t getTensorElementCount(llvm::Type* llvmType) {
    if (llvmType->isArrayTy()) {
        auto arrayType = llvm::cast<llvm::ArrayType>(llvmType);
        return arrayType->getNumElements() * 
               getTensorElementCount(arrayType->getElementType());
    } else if (llvmType->isVectorTy()) {
        if (auto fixedVectorType = llvm::dyn_cast<llvm::FixedVectorType>(llvmType)) {
            return fixedVectorType->getNumElements();
        }
        return 1; // Fallback for scalable vectors
    } else {
        return 1; // Scalar types
    }
}

} // namespace utils

} // namespace llvm2tosa