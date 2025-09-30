#include "LLVMToTosaConverter.h"
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/BasicBlock.h>
#include <mlir/IR/MLIRContext.h>
#include <iostream>

int main() {
    // Initialize LLVM context and create a simple module
    llvm::LLVMContext llvmContext;
    auto llvmModule = std::make_unique<llvm::Module>("test_module", llvmContext);
    
    // Create a simple function: int add(int a, int b) { return a + b; }
    llvm::FunctionType* funcType = llvm::FunctionType::get(
        llvm::Type::getInt32Ty(llvmContext),
        {llvm::Type::getInt32Ty(llvmContext), llvm::Type::getInt32Ty(llvmContext)},
        false
    );
    
    llvm::Function* func = llvm::Function::Create(
        funcType, llvm::Function::ExternalLinkage, "add", llvmModule.get()
    );
    
    llvm::BasicBlock* entry = llvm::BasicBlock::Create(llvmContext, "entry", func);
    llvm::IRBuilder<> builder(entry);
    
    auto args = func->args();
    auto a = args.begin();
    auto b = std::next(args.begin());
    
    llvm::Value* sum = builder.CreateAdd(a, b, "sum");
    builder.CreateRet(sum);
    
    // Initialize MLIR context and converter
    mlir::MLIRContext mlirContext;
    llvm2tosa::LLVMToTosaConverter converter(mlirContext);
    
    try {
        // Convert the LLVM module to TOSA
        auto tosaModule = converter.convertModule(*llvmModule);
        
        std::cout << "Conversion successful!" << std::endl;
        std::cout << "TOSA module created with " << tosaModule->getBody()->getOperations().size() 
                  << " operations" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Conversion failed: " << e.what() << std::endl;
        return 1;
    }
}