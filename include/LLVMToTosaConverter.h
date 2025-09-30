#pragma once

#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/IRBuilder.h>

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Arith/IR/Arith.h>

#include <map>
#include <vector>
#include <string>
#include <memory>

namespace llvm2tosa {

// Forward declarations
class TypeConverter;
class MemoryModelConverter;
class ControlFlowConverter;
class InstructionConverter;

/**
 * @brief Comprehensive LLVM IR to TOSA IR Converter
 * 
 * This converter handles the complete transformation from LLVM IR to TOSA IR,
 * addressing fundamental differences between:
 * - Scalar/pointer-based computation (LLVM) vs tensor-based computation (TOSA)
 * - Explicit memory model (LLVM) vs immutable value semantics (TOSA)
 * - Basic block CFG (LLVM) vs structured control flow (TOSA)
 */
class LLVMToTosaConverter {
public:
    LLVMToTosaConverter(mlir::MLIRContext& context);
    ~LLVMToTosaConverter();

    /**
     * @brief Convert an entire LLVM module to TOSA IR
     */
    std::unique_ptr<mlir::ModuleOp> convertModule(llvm::Module& llvmModule);

    /**
     * @brief Convert a single LLVM function to TOSA function
     */
    mlir::func::FuncOp convertFunction(llvm::Function& llvmFunc);

    /**
     * @brief Set conversion options
     */
    void setQuantizationMode(bool enable) { enableQuantization_ = enable; }
    void setOptimizationLevel(int level) { optimizationLevel_ = level; }
    void setDebugMode(bool enable) { debugMode_ = enable; }

private:
    mlir::MLIRContext& context_;
    mlir::OpBuilder builder_;
    
    // Sub-converters for different aspects
    std::unique_ptr<TypeConverter> typeConverter_;
    std::unique_ptr<MemoryModelConverter> memoryConverter_;
    std::unique_ptr<ControlFlowConverter> controlFlowConverter_;
    std::unique_ptr<InstructionConverter> instructionConverter_;
    
    // Configuration options
    bool enableQuantization_;
    int optimizationLevel_;
    bool debugMode_;
    
    // State tracking
    std::map<llvm::Value*, mlir::Value> valueMapping_;
    std::map<llvm::BasicBlock*, mlir::Block*> blockMapping_;
    std::map<llvm::Function*, mlir::func::FuncOp> functionMapping_;
    
    // Helper methods
    void initializeSubConverters();
    void processGlobalVariables(llvm::Module& llvmModule, mlir::ModuleOp& tosaModule);
    void processFunctionDeclarations(llvm::Module& llvmModule, mlir::ModuleOp& tosaModule);
    void convertBasicBlock(llvm::BasicBlock& bb, mlir::func::FuncOp& tosaFunc);
};

/**
 * @brief Handles type conversion between LLVM and TOSA type systems
 */
class TypeConverter {
public:
    TypeConverter(mlir::MLIRContext& context) : context_(context) {}

    /**
     * @brief Convert LLVM type to TOSA tensor type
     */
    mlir::Type convertType(llvm::Type* llvmType);
    
    /**
     * @brief Convert function signature
     */
    mlir::FunctionType convertFunctionType(llvm::FunctionType* llvmFuncType);

private:
    mlir::MLIRContext& context_;
    
    // Type mapping cache
    std::map<llvm::Type*, mlir::Type> typeCache_;
    
    // Helper methods for specific type conversions
    mlir::Type convertIntegerType(llvm::IntegerType* intType);
    mlir::Type convertFloatingPointType(llvm::Type* fpType);
    mlir::Type convertPointerType(llvm::PointerType* ptrType);
    mlir::Type convertArrayType(llvm::ArrayType* arrayType);
    mlir::Type convertVectorType(llvm::VectorType* vectorType);
    mlir::Type convertStructType(llvm::StructType* structType);
    
    // Tensor shape inference
    std::vector<int64_t> inferTensorShape(llvm::Type* llvmType);
    mlir::Type getElementType(llvm::Type* llvmType);
};

/**
 * @brief Converts LLVM's explicit memory model to TOSA's tensor operations
 */
class MemoryModelConverter {
public:
    MemoryModelConverter(mlir::MLIRContext& context, TypeConverter& typeConverter);

    /**
     * @brief Convert alloca instruction to tensor initialization
     */
    mlir::Value convertAlloca(llvm::AllocaInst* allocaInst, mlir::OpBuilder& builder);
    
    /**
     * @brief Convert load instruction to tensor slice operation
     */
    mlir::Value convertLoad(llvm::LoadInst* loadInst, mlir::OpBuilder& builder);
    
    /**
     * @brief Convert store instruction to tensor update operation
     */
    void convertStore(llvm::StoreInst* storeInst, mlir::OpBuilder& builder);
    
    /**
     * @brief Convert getelementptr to tensor indexing
     */
    mlir::Value convertGEP(llvm::GetElementPtrInst* gepInst, mlir::OpBuilder& builder);

    /**
     * @brief Handle global variables conversion
     */
    mlir::Value convertGlobalVariable(llvm::GlobalVariable* globalVar, mlir::OpBuilder& builder);

private:
    mlir::MLIRContext& context_;
    TypeConverter& typeConverter_;
    
    // Memory allocation tracking
    struct MemoryAllocation {
        mlir::Type tensorType;
        llvm::Type* originalType;
        std::vector<int64_t> shape;
        bool isGlobal;
    };
    
    std::map<llvm::Value*, MemoryAllocation> memoryAllocations_;
    std::map<llvm::Value*, mlir::Value> tensorValues_;
    
    // Helper methods
    mlir::Value createZeroTensor(mlir::Type tensorType, mlir::OpBuilder& builder);
    mlir::Value createConstantTensor(llvm::Constant* constant, mlir::OpBuilder& builder);
    std::vector<mlir::Value> computeIndices(llvm::GetElementPtrInst* gepInst, mlir::OpBuilder& builder);
    mlir::Value updateTensorAtIndices(mlir::Value tensor, mlir::Value value, 
                                     const std::vector<mlir::Value>& indices, 
                                     mlir::OpBuilder& builder);
};

/**
 * @brief Converts LLVM instructions to TOSA operations
 */
class InstructionConverter {
public:
    InstructionConverter(mlir::MLIRContext& context, TypeConverter& typeConverter,
                        MemoryModelConverter& memoryConverter);

    /**
     * @brief Convert a single LLVM instruction to TOSA operations
     */
    mlir::Value convertInstruction(llvm::Instruction* inst, mlir::OpBuilder& builder);

private:
    mlir::MLIRContext& context_;
    TypeConverter& typeConverter_;
    MemoryModelConverter& memoryConverter_;
    
    // Instruction conversion methods for each LLVM instruction type
    
    // Arithmetic instructions
    mlir::Value convertAdd(llvm::BinaryOperator* inst, mlir::OpBuilder& builder);
    mlir::Value convertSub(llvm::BinaryOperator* inst, mlir::OpBuilder& builder);
    mlir::Value convertMul(llvm::BinaryOperator* inst, mlir::OpBuilder& builder);
    mlir::Value convertUDiv(llvm::BinaryOperator* inst, mlir::OpBuilder& builder);
    mlir::Value convertSDiv(llvm::BinaryOperator* inst, mlir::OpBuilder& builder);
    mlir::Value convertURem(llvm::BinaryOperator* inst, mlir::OpBuilder& builder);
    mlir::Value convertSRem(llvm::BinaryOperator* inst, mlir::OpBuilder& builder);
    
    // Floating-point arithmetic
    mlir::Value convertFAdd(llvm::BinaryOperator* inst, mlir::OpBuilder& builder);
    mlir::Value convertFSub(llvm::BinaryOperator* inst, mlir::OpBuilder& builder);
    mlir::Value convertFMul(llvm::BinaryOperator* inst, mlir::OpBuilder& builder);
    mlir::Value convertFDiv(llvm::BinaryOperator* inst, mlir::OpBuilder& builder);
    mlir::Value convertFRem(llvm::BinaryOperator* inst, mlir::OpBuilder& builder);
    
    // Bitwise operations
    mlir::Value convertAnd(llvm::BinaryOperator* inst, mlir::OpBuilder& builder);
    mlir::Value convertOr(llvm::BinaryOperator* inst, mlir::OpBuilder& builder);
    mlir::Value convertXor(llvm::BinaryOperator* inst, mlir::OpBuilder& builder);
    mlir::Value convertShl(llvm::BinaryOperator* inst, mlir::OpBuilder& builder);
    mlir::Value convertLShr(llvm::BinaryOperator* inst, mlir::OpBuilder& builder);
    mlir::Value convertAShr(llvm::BinaryOperator* inst, mlir::OpBuilder& builder);
    
    // Comparison operations
    mlir::Value convertICmp(llvm::ICmpInst* inst, mlir::OpBuilder& builder);
    mlir::Value convertFCmp(llvm::FCmpInst* inst, mlir::OpBuilder& builder);
    
    // Conversion operations
    mlir::Value convertTrunc(llvm::TruncInst* inst, mlir::OpBuilder& builder);
    mlir::Value convertZExt(llvm::ZExtInst* inst, mlir::OpBuilder& builder);
    mlir::Value convertSExt(llvm::SExtInst* inst, mlir::OpBuilder& builder);
    mlir::Value convertFPTrunc(llvm::FPTruncInst* inst, mlir::OpBuilder& builder);
    mlir::Value convertFPExt(llvm::FPExtInst* inst, mlir::OpBuilder& builder);
    mlir::Value convertFPToUI(llvm::FPToUIInst* inst, mlir::OpBuilder& builder);
    mlir::Value convertFPToSI(llvm::FPToSIInst* inst, mlir::OpBuilder& builder);
    mlir::Value convertUIToFP(llvm::UIToFPInst* inst, mlir::OpBuilder& builder);
    mlir::Value convertSIToFP(llvm::SIToFPInst* inst, mlir::OpBuilder& builder);
    mlir::Value convertPtrToInt(llvm::PtrToIntInst* inst, mlir::OpBuilder& builder);
    mlir::Value convertIntToPtr(llvm::IntToPtrInst* inst, mlir::OpBuilder& builder);
    mlir::Value convertBitCast(llvm::BitCastInst* inst, mlir::OpBuilder& builder);
    
    // Vector operations
    mlir::Value convertExtractElement(llvm::ExtractElementInst* inst, mlir::OpBuilder& builder);
    mlir::Value convertInsertElement(llvm::InsertElementInst* inst, mlir::OpBuilder& builder);
    mlir::Value convertShuffleVector(llvm::ShuffleVectorInst* inst, mlir::OpBuilder& builder);
    
    // Other operations
    mlir::Value convertSelect(llvm::SelectInst* inst, mlir::OpBuilder& builder);
    mlir::Value convertPHI(llvm::PHINode* inst, mlir::OpBuilder& builder);
    mlir::Value convertCall(llvm::CallInst* inst, mlir::OpBuilder& builder);
    
    // Intrinsic functions
    mlir::Value convertIntrinsic(llvm::IntrinsicInst* inst, mlir::OpBuilder& builder);
    mlir::Value convertMemCpy(llvm::MemCpyInst* inst, mlir::OpBuilder& builder);
    mlir::Value convertMemMove(llvm::MemMoveInst* inst, mlir::OpBuilder& builder);
    mlir::Value convertMemSet(llvm::MemSetInst* inst, mlir::OpBuilder& builder);
    
    // Helper methods
    std::pair<mlir::Value, mlir::Value> broadcastToCompatibleShape(mlir::Value lhs, mlir::Value rhs, mlir::OpBuilder& builder);
    mlir::Value ensureTensorType(mlir::Value value, mlir::OpBuilder& builder);
    mlir::Value createQuantizationShift(mlir::OpBuilder& builder);
    mlir::Value getConvertedValue(llvm::Value* llvmValue, mlir::OpBuilder& builder);
    mlir::Value convertConstant(llvm::Constant* constant, mlir::OpBuilder& builder);
    mlir::Value createZeroTensor(mlir::Type tensorType, mlir::OpBuilder& builder);
    
private:
    mlir::MLIRContext& context_;
    TypeConverter& typeConverter_;
    MemoryModelConverter& memoryConverter_;
    std::map<llvm::Value*, mlir::Value> valueMapping_;
};

/**
 * @brief Converts LLVM control flow to TOSA structured control flow
 */
class ControlFlowConverter {
public:
    ControlFlowConverter(mlir::MLIRContext& context, TypeConverter& typeConverter);

    /**
     * @brief Analyze and convert control flow structure
     */
    void convertControlFlow(llvm::Function& llvmFunc, mlir::func::FuncOp& tosaFunc);

private:
    mlir::MLIRContext& context_;
    TypeConverter& typeConverter_;
    
    // Control flow analysis structures
    struct LoopInfo {
        llvm::BasicBlock* header;
        llvm::BasicBlock* latch;
        std::vector<llvm::BasicBlock*> blocks;
        bool isNaturalLoop;
    };
    
    struct IfInfo {
        llvm::BasicBlock* condition;
        llvm::BasicBlock* thenBlock;
        llvm::BasicBlock* elseBlock;
        llvm::BasicBlock* merge;
    };
    
    std::vector<LoopInfo> loops_;
    std::vector<IfInfo> conditionals_;
    
    // Analysis methods
    void analyzeControlFlow(llvm::Function& llvmFunc);
    void identifyLoops(llvm::Function& llvmFunc);
    void identifyConditionals(llvm::Function& llvmFunc);
    
    // Conversion methods
    void convertLoop(const LoopInfo& loop, mlir::OpBuilder& builder);
    void convertConditional(const IfInfo& conditional, mlir::OpBuilder& builder);
    void convertBasicBlock(llvm::BasicBlock& bb, mlir::OpBuilder& builder);
    
    // Terminator instruction handlers
    void convertReturn(llvm::ReturnInst* ret, mlir::OpBuilder& builder);
    void convertBranch(llvm::BranchInst* br, mlir::OpBuilder& builder);
    void convertSwitch(llvm::SwitchInst* sw, mlir::OpBuilder& builder);
    void convertIndirectBr(llvm::IndirectBrInst* ibr, mlir::OpBuilder& builder);
    void convertInvoke(llvm::InvokeInst* invoke, mlir::OpBuilder& builder);
    void convertUnreachable(llvm::UnreachableInst* unreachable, mlir::OpBuilder& builder);
    
    // Helper methods
    bool isNaturalLoop(const std::vector<llvm::BasicBlock*>& blocks);
    llvm::BasicBlock* findLoopLatch(llvm::BasicBlock* header);
    mlir::Value convertCondition(llvm::Value* condition, mlir::OpBuilder& builder);
};

/**
 * @brief Utility functions for the converter
 */
namespace utils {
    /**
     * @brief Create MLIR location from LLVM instruction
     */
    mlir::Location createLocation(llvm::Instruction* inst, mlir::MLIRContext& context);
    
    /**
     * @brief Check if LLVM type can be directly converted to tensor
     */
    bool isDirectlyConvertible(llvm::Type* llvmType);
    
    /**
     * @brief Get tensor element count from LLVM type
     */
    size_t getTensorElementCount(llvm::Type* llvmType);
    
    /**
     * @brief Create debug information for converted operations
     */
    void attachDebugInfo(mlir::Operation* op, llvm::Instruction* inst);
}

} // namespace llvm2tosa