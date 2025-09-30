#include "LLVMToTosaConverter.h"
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/IR/Builders.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalVariable.h>

namespace llvm2tosa {

MemoryModelConverter::MemoryModelConverter(mlir::MLIRContext& context, TypeConverter& typeConverter)
    : context_(context), typeConverter_(typeConverter) {}

mlir::Value MemoryModelConverter::convertAlloca(llvm::AllocaInst* allocaInst, mlir::OpBuilder& builder) {
    // Convert LLVM alloca to TOSA tensor initialization
    llvm::Type* allocatedType = allocaInst->getAllocatedType();
    mlir::Type tensorType = typeConverter_.convertType(allocatedType);
    
    // Track this allocation
    MemoryAllocation allocation;
    allocation.tensorType = tensorType;
    allocation.originalType = allocatedType;
    allocation.isGlobal = false;
    
    if (auto rankedTensorType = tensorType.dyn_cast<mlir::RankedTensorType>()) {
        allocation.shape = rankedTensorType.getShape().vec();
    } else {
        allocation.shape = {1}; // Default to scalar
    }
    
    memoryAllocations_[allocaInst] = allocation;
    
    // Create zero-initialized tensor
    mlir::Value tensorValue = createZeroTensor(tensorType, builder);
    tensorValues_[allocaInst] = tensorValue;
    
    return tensorValue;
}

mlir::Value MemoryModelConverter::convertLoad(llvm::LoadInst* loadInst, mlir::OpBuilder& builder) {
    llvm::Value* pointer = loadInst->getPointerOperand();
    
    // Find the tensor corresponding to this pointer
    auto it = tensorValues_.find(pointer);
    if (it != tensorValues_.end()) {
        // Direct tensor access
        return it->second;
    }
    
    // Handle GEP-based access
    if (auto gepInst = llvm::dyn_cast<llvm::GetElementPtrInst>(pointer)) {
        mlir::Value baseValue = convertGEP(gepInst, builder);
        return baseValue;
    }
    
    // Handle global variable access
    if (auto globalVar = llvm::dyn_cast<llvm::GlobalVariable>(pointer)) {
        return convertGlobalVariable(globalVar, builder);
    }
    
    // Fallback: create a placeholder tensor
    mlir::Type resultType = typeConverter_.convertType(loadInst->getType());
    return createZeroTensor(resultType, builder);
}

void MemoryModelConverter::convertStore(llvm::StoreInst* storeInst, mlir::OpBuilder& builder) {
    llvm::Value* valueToStore = storeInst->getValueOperand();
    llvm::Value* pointer = storeInst->getPointerOperand();
    
    // Convert the value to be stored
    mlir::Value storeValue;
    
    // Handle constants
    if (auto constant = llvm::dyn_cast<llvm::Constant>(valueToStore)) {
        storeValue = createConstantTensor(constant, builder);
    } else {
        // The value should already be converted
        auto it = tensorValues_.find(valueToStore);
        if (it != tensorValues_.end()) {
            storeValue = it->second;
        } else {
            // Create a placeholder
            mlir::Type valueType = typeConverter_.convertType(valueToStore->getType());
            storeValue = createZeroTensor(valueType, builder);
        }
    }
    
    // Handle different pointer types
    if (auto gepInst = llvm::dyn_cast<llvm::GetElementPtrInst>(pointer)) {
        // GEP-based store: update tensor at specific indices
        llvm::Value* basePointer = gepInst->getPointerOperand();
        auto baseIt = tensorValues_.find(basePointer);
        if (baseIt != tensorValues_.end()) {
            std::vector<mlir::Value> indices = computeIndices(gepInst, builder);
            mlir::Value updatedTensor = updateTensorAtIndices(baseIt->second, storeValue, indices, builder);
            tensorValues_[basePointer] = updatedTensor;
        }
    } else {
        // Direct store: replace entire tensor
        tensorValues_[pointer] = storeValue;
    }
}

mlir::Value MemoryModelConverter::convertGEP(llvm::GetElementPtrInst* gepInst, mlir::OpBuilder& builder) {
    llvm::Value* basePointer = gepInst->getPointerOperand();
    
    // Find base tensor
    auto baseIt = tensorValues_.find(basePointer);
    if (baseIt == tensorValues_.end()) {
        // Create a placeholder tensor
        mlir::Type baseType = typeConverter_.convertType(gepInst->getSourceElementType());
        return createZeroTensor(baseType, builder);
    }
    
    mlir::Value baseTensor = baseIt->second;
    
    // Compute indices
    std::vector<mlir::Value> indices = computeIndices(gepInst, builder);
    
    // Extract slice from tensor
    if (indices.empty()) {
        return baseTensor;
    }
    
    // Create slice operation
    auto tensorType = baseTensor.getType().cast<mlir::RankedTensorType>();
    std::vector<int64_t> starts(tensorType.getRank(), 0);
    std::vector<int64_t> sizes = tensorType.getShape().vec();
    
    // Apply computed indices
    for (size_t i = 0; i < std::min(indices.size(), starts.size()); ++i) {
        if (auto constIndex = indices[i].getDefiningOp<mlir::arith::ConstantIndexOp>()) {
            starts[i] = constIndex.value();
            sizes[i] = 1;
        }
    }
    
    auto startAttr = builder.getI64ArrayAttr(starts);
    auto sizeAttr = builder.getI64ArrayAttr(sizes);
    
    // Determine result type
    mlir::Type resultType = typeConverter_.convertType(gepInst->getResultElementType());
    if (auto rankedType = resultType.dyn_cast<mlir::RankedTensorType>()) {
        // Use the size as the shape
        resultType = mlir::RankedTensorType::get(sizes, rankedType.getElementType());
    }
    
    return builder.create<mlir::tosa::SliceOp>(
        builder.getUnknownLoc(), resultType, baseTensor, startAttr, sizeAttr);
}

mlir::Value MemoryModelConverter::convertGlobalVariable(llvm::GlobalVariable* globalVar, mlir::OpBuilder& builder) {
    // Convert global variable to constant tensor
    if (globalVar->hasInitializer()) {
        llvm::Constant* initializer = globalVar->getInitializer();
        return createConstantTensor(initializer, builder);
    } else {
        // Uninitialized global: create zero tensor
        mlir::Type globalType = typeConverter_.convertType(globalVar->getValueType());
        return createZeroTensor(globalType, builder);
    }
}

mlir::Value MemoryModelConverter::createZeroTensor(mlir::Type tensorType, mlir::OpBuilder& builder) {
    auto rankedTensorType = tensorType.cast<mlir::RankedTensorType>();
    
    mlir::Attribute zeroAttr;
    if (rankedTensorType.getElementType().isa<mlir::IntegerType>()) {
        zeroAttr = mlir::DenseElementsAttr::get(rankedTensorType, 0);
    } else if (rankedTensorType.getElementType().isa<mlir::FloatType>()) {
        zeroAttr = mlir::DenseElementsAttr::get(rankedTensorType, 0.0f);
    } else {
        // Fallback to integer zero
        auto intType = mlir::IntegerType::get(&context_, 32);
        auto intTensorType = mlir::RankedTensorType::get(rankedTensorType.getShape(), intType);
        zeroAttr = mlir::DenseElementsAttr::get(intTensorType, 0);
        rankedTensorType = intTensorType;
    }
    
    return builder.create<mlir::tosa::ConstOp>(
        builder.getUnknownLoc(), rankedTensorType, zeroAttr);
}

mlir::Value MemoryModelConverter::createConstantTensor(llvm::Constant* constant, mlir::OpBuilder& builder) {
    mlir::Type tensorType = typeConverter_.convertType(constant->getType());
    auto rankedTensorType = tensorType.cast<mlir::RankedTensorType>();
    
    mlir::Attribute constAttr;
    
    if (auto constInt = llvm::dyn_cast<llvm::ConstantInt>(constant)) {
        // Integer constant
        int64_t value = constInt->getSExtValue();
        constAttr = mlir::DenseElementsAttr::get(rankedTensorType, value);
    } else if (auto constFP = llvm::dyn_cast<llvm::ConstantFP>(constant)) {
        // Floating-point constant
        double value = constFP->getValueAPF().convertToDouble();
        constAttr = mlir::DenseElementsAttr::get(rankedTensorType, value);
    } else if (auto constArray = llvm::dyn_cast<llvm::ConstantArray>(constant)) {
        // Array constant
        std::vector<mlir::Attribute> elements;
        for (unsigned i = 0; i < constArray->getNumOperands(); ++i) {
            llvm::Constant* element = constArray->getOperand(i);
            if (auto elementInt = llvm::dyn_cast<llvm::ConstantInt>(element)) {
                elements.push_back(builder.getIntegerAttr(
                    rankedTensorType.getElementType(), elementInt->getSExtValue()));
            } else if (auto elementFP = llvm::dyn_cast<llvm::ConstantFP>(element)) {
                elements.push_back(builder.getFloatAttr(
                    rankedTensorType.getElementType(), elementFP->getValueAPF().convertToDouble()));
            }
        }
        constAttr = mlir::DenseElementsAttr::get(rankedTensorType, elements);
    } else if (auto constVector = llvm::dyn_cast<llvm::ConstantVector>(constant)) {
        // Vector constant
        std::vector<mlir::Attribute> elements;
        for (unsigned i = 0; i < constVector->getNumOperands(); ++i) {
            llvm::Constant* element = constVector->getOperand(i);
            if (auto elementInt = llvm::dyn_cast<llvm::ConstantInt>(element)) {
                elements.push_back(builder.getIntegerAttr(
                    rankedTensorType.getElementType(), elementInt->getSExtValue()));
            } else if (auto elementFP = llvm::dyn_cast<llvm::ConstantFP>(element)) {
                elements.push_back(builder.getFloatAttr(
                    rankedTensorType.getElementType(), elementFP->getValueAPF().convertToDouble()));
            }
        }
        constAttr = mlir::DenseElementsAttr::get(rankedTensorType, elements);
    } else {
        // Fallback: zero tensor
        return createZeroTensor(tensorType, builder);
    }
    
    return builder.create<mlir::tosa::ConstOp>(
        builder.getUnknownLoc(), rankedTensorType, constAttr);
}

std::vector<mlir::Value> MemoryModelConverter::computeIndices(llvm::GetElementPtrInst* gepInst, 
                                                              mlir::OpBuilder& builder) {
    std::vector<mlir::Value> indices;
    
    // Skip the first index (it's usually 0 for the base pointer)
    for (unsigned i = 1; i < gepInst->getNumIndices(); ++i) {
        llvm::Value* index = gepInst->getOperand(i + 1); // +1 because first operand is pointer
        
        if (auto constIndex = llvm::dyn_cast<llvm::ConstantInt>(index)) {
            // Constant index
            mlir::Value indexValue = builder.create<mlir::arith::ConstantIndexOp>(
                builder.getUnknownLoc(), constIndex->getSExtValue());
            indices.push_back(indexValue);
        } else {
            // Variable index - would need to be converted from LLVM value
            // For now, use zero as placeholder
            mlir::Value indexValue = builder.create<mlir::arith::ConstantIndexOp>(
                builder.getUnknownLoc(), 0);
            indices.push_back(indexValue);
        }
    }
    
    return indices;
}

mlir::Value MemoryModelConverter::updateTensorAtIndices(mlir::Value tensor, mlir::Value value,
                                                        const std::vector<mlir::Value>& indices,
                                                        mlir::OpBuilder& builder) {
    // For now, implement a simplified update using slice and concat operations
    // A more sophisticated implementation would use scatter operations or
    // convert to multiple slice/concat operations
    
    auto tensorType = tensor.getType().cast<mlir::RankedTensorType>();
    
    if (indices.empty()) {
        return value;
    }
    
    // For 1D case, we can use slice and concat
    if (tensorType.getRank() == 1 && indices.size() == 1) {
        // Get the index value
        auto indexOp = indices[0].getDefiningOp<mlir::arith::ConstantIndexOp>();
        if (!indexOp) {
            // Dynamic index - more complex, return original tensor for now
            return tensor;
        }
        
        int64_t index = indexOp.value();
        int64_t tensorSize = tensorType.getShape()[0];
        
        if (index < 0 || index >= tensorSize) {
            return tensor; // Out of bounds
        }
        
        // Create prefix slice (0 to index)
        mlir::Value prefix;
        if (index > 0) {
            auto prefixStartAttr = builder.getI64ArrayAttr({0});
            auto prefixSizeAttr = builder.getI64ArrayAttr({index});
            auto prefixType = mlir::RankedTensorType::get({index}, tensorType.getElementType());
            prefix = builder.create<mlir::tosa::SliceOp>(
                builder.getUnknownLoc(), prefixType, tensor, prefixStartAttr, prefixSizeAttr);
        }
        
        // Create suffix slice (index+1 to end)
        mlir::Value suffix;
        if (index < tensorSize - 1) {
            auto suffixStartAttr = builder.getI64ArrayAttr({index + 1});
            auto suffixSizeAttr = builder.getI64ArrayAttr({tensorSize - index - 1});
            auto suffixType = mlir::RankedTensorType::get({tensorSize - index - 1}, tensorType.getElementType());
            suffix = builder.create<mlir::tosa::SliceOp>(
                builder.getUnknownLoc(), suffixType, tensor, suffixStartAttr, suffixSizeAttr);
        }
        
        // Concatenate prefix + value + suffix
        llvm::SmallVector<mlir::Value> concatOperands;
        if (prefix) concatOperands.push_back(prefix);
        concatOperands.push_back(value);
        if (suffix) concatOperands.push_back(suffix);
        
        if (concatOperands.size() == 1) {
            return concatOperands[0];
        }
        
        return builder.create<mlir::tosa::ConcatOp>(
            builder.getUnknownLoc(), tensorType, concatOperands, builder.getI32IntegerAttr(0));
    }
    
    // For multi-dimensional tensors, return original for now
    // A complete implementation would handle this recursively
    return tensor;
}

} // namespace llvm2tosa