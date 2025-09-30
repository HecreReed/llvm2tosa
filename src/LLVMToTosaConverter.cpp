#include "LLVMToTosaConverter.h"
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Instructions.h>

namespace llvm2tosa {

LLVMToTosaConverter::LLVMToTosaConverter(mlir::MLIRContext& context)
    : context_(context), builder_(&context_), enableQuantization_(false), 
      optimizationLevel_(0), debugMode_(false) {
    initializeSubConverters();
}

LLVMToTosaConverter::~LLVMToTosaConverter() = default;

void LLVMToTosaConverter::initializeSubConverters() {
    typeConverter_ = std::make_unique<TypeConverter>(context_);
    memoryConverter_ = std::make_unique<MemoryModelConverter>(context_, *typeConverter_);
    controlFlowConverter_ = std::make_unique<ControlFlowConverter>(context_, *typeConverter_);
    instructionConverter_ = std::make_unique<InstructionConverter>(context_, *typeConverter_, *memoryConverter_);
}

std::unique_ptr<mlir::ModuleOp> LLVMToTosaConverter::convertModule(llvm::Module& llvmModule) {
    // Create MLIR module
    auto tosaModule = mlir::ModuleOp::create(builder_.getUnknownLoc());
    builder_.setInsertionPointToEnd(tosaModule.getBody());
    
    // Process global variables first
    processGlobalVariables(llvmModule, *tosaModule);
    
    // Process function declarations
    processFunctionDeclarations(llvmModule, *tosaModule);
    
    // Convert each function
    for (auto& llvmFunc : llvmModule) {
        if (!llvmFunc.isDeclaration()) {
            mlir::func::FuncOp tosaFunc = convertFunction(llvmFunc);
            functionMapping_[&llvmFunc] = tosaFunc;
        }
    }
    
    return std::make_unique<mlir::ModuleOp>(tosaModule);
}

mlir::func::FuncOp LLVMToTosaConverter::convertFunction(llvm::Function& llvmFunc) {
    // Convert function type
    mlir::FunctionType funcType = typeConverter_->convertFunctionType(llvmFunc.getFunctionType());
    
    // Create MLIR function
    auto tosaFunc = builder_.create<mlir::func::FuncOp>(
        utils::createLocation(&llvmFunc.front().front(), context_),
        llvmFunc.getName(),
        funcType
    );
    
    // If the function is a declaration, we're done
    if (llvmFunc.isDeclaration()) {
        return tosaFunc;
    }
    
    // Create function body
    mlir::Block* entryBlock = tosaFunc.addEntryBlock();
    builder_.setInsertionPointToEnd(entryBlock);
    
    // Map function arguments
    auto llvmArgs = llvmFunc.args();
    auto mlirArgs = entryBlock->getArguments();
    auto llvmArgIt = llvmArgs.begin();
    auto mlirArgIt = mlirArgs.begin();
    
    while (llvmArgIt != llvmArgs.end() && mlirArgIt != mlirArgs.end()) {
        valueMapping_[&*llvmArgIt] = *mlirArgIt;
        ++llvmArgIt;
        ++mlirArgIt;
    }
    
    // Clear previous mappings for this function
    blockMapping_.clear();
    
    // Create blocks for all basic blocks first
    for (auto& bb : llvmFunc) {
        mlir::Block* mlirBlock;
        if (&bb == &llvmFunc.getEntryBlock()) {
            mlirBlock = entryBlock;
        } else {
            mlirBlock = builder_.createBlock(&tosaFunc.getBody());
        }
        blockMapping_[&bb] = mlirBlock;
    }
    
    // Convert basic blocks
    for (auto& bb : llvmFunc) {
        convertBasicBlock(bb, tosaFunc);
    }
    
    // Handle control flow conversion
    controlFlowConverter_->convertControlFlow(llvmFunc, tosaFunc);
    
    return tosaFunc;
}

void LLVMToTosaConverter::convertBasicBlock(llvm::BasicBlock& bb, mlir::func::FuncOp& tosaFunc) {
    mlir::Block* mlirBlock = blockMapping_[&bb];
    builder_.setInsertionPointToEnd(mlirBlock);
    
    // Convert each instruction in the basic block
    for (auto& inst : bb) {
        mlir::Value result = instructionConverter_->convertInstruction(&inst, builder_);
        
        // Map the result if the instruction produces a value
        if (!inst.getType()->isVoidTy() && result) {
            valueMapping_[&inst] = result;
        }
    }
}

void LLVMToTosaConverter::processGlobalVariables(llvm::Module& llvmModule, mlir::ModuleOp& tosaModule) {
    builder_.setInsertionPointToEnd(tosaModule.getBody());
    
    for (auto& globalVar : llvmModule.globals()) {
        // Convert global variable to module-level constant or variable
        mlir::Type globalType = typeConverter_->convertType(globalVar.getValueType());
        
        if (globalVar.hasInitializer()) {
            // Create constant global
            llvm::Constant* initializer = globalVar.getInitializer();
            
            // For now, create a simple constant operation
            // A complete implementation would handle complex initializers
            if (auto rankedType = globalType.dyn_cast<mlir::RankedTensorType>()) {
                mlir::Attribute constAttr;
                
                if (auto constInt = llvm::dyn_cast<llvm::ConstantInt>(initializer)) {
                    constAttr = mlir::DenseElementsAttr::get(rankedType, constInt->getSExtValue());
                } else if (auto constFP = llvm::dyn_cast<llvm::ConstantFP>(initializer)) {
                    constAttr = mlir::DenseElementsAttr::get(rankedType, constFP->getValueAPF().convertToDouble());
                } else {
                    // Default to zero
                    constAttr = mlir::DenseElementsAttr::get(rankedType, 0);
                }
                
                auto globalConstOp = builder_.create<mlir::tosa::ConstOp>(
                    utils::createLocation(nullptr, context_), rankedType, constAttr);
                
                // Store in a global symbol table or similar mechanism
                // For now, just track the conversion
                valueMapping_[&globalVar] = globalConstOp.getResult();
            }
        }
    }
}

void LLVMToTosaConverter::processFunctionDeclarations(llvm::Module& llvmModule, mlir::ModuleOp& tosaModule) {
    builder_.setInsertionPointToEnd(tosaModule.getBody());
    
    // Create function declarations for external functions
    for (auto& llvmFunc : llvmModule) {
        if (llvmFunc.isDeclaration()) {
            mlir::FunctionType funcType = typeConverter_->convertFunctionType(llvmFunc.getFunctionType());
            
            auto tosaFuncDecl = builder_.create<mlir::func::FuncOp>(
                utils::createLocation(nullptr, context_),
                llvmFunc.getName(),
                funcType
            );
            
            // Mark as declaration
            tosaFuncDecl.setPrivate();
            
            functionMapping_[&llvmFunc] = tosaFuncDecl;
        }
    }
}

// Utility functions implementation
namespace utils {

mlir::Location createLocation(llvm::Instruction* inst, mlir::MLIRContext& context) {
    if (inst && inst->getDebugLoc()) {
        // Get debug location information
        auto debugLoc = inst->getDebugLoc();
        unsigned line = debugLoc.getLine();
        unsigned col = debugLoc.getCol();
        
        // Create file location
        auto filename = debugLoc->getFilename();
        auto fileAttr = mlir::StringAttr::get(&context, filename);
        return mlir::FileLineColLoc::get(fileAttr, line, col);
    }
    
    // Fallback to unknown location
    return mlir::UnknownLoc::get(&context);
}

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

void attachDebugInfo(mlir::Operation* op, llvm::Instruction* inst) {
    if (inst && inst->getDebugLoc()) {
        auto debugLoc = inst->getDebugLoc();
        unsigned line = debugLoc.getLine();
        unsigned col = debugLoc.getCol();
        auto filename = debugLoc->getFilename();
        
        auto& context = op->getContext();
        auto fileAttr = mlir::StringAttr::get(&context, filename);
        auto loc = mlir::FileLineColLoc::get(fileAttr, line, col);
        op->setLoc(loc);
    }
}

} // namespace utils

} // namespace llvm2tosa