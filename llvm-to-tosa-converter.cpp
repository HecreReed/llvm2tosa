#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace mlir::tosa;

namespace {

// LLVM IR to TOSA conversion patterns
class LLVMToTosaTypeConverter : public TypeConverter {
public:
  LLVMToTosaTypeConverter() {
    // Convert LLVM types to tensor types
    addConversion([](Type type) { return type; });
    
    // Convert LLVM integer types to tensor<i32>
    addConversion([](LLVM::LLVMPointerType type) -> std::optional<Type> {
      auto ctx = type.getContext();
      return RankedTensorType::get({1}, IntegerType::get(ctx, 32));
    });
    
    // Convert LLVM array types to tensors
    addConversion([](LLVM::LLVMArrayType arrayType) -> std::optional<Type> {
      auto elementType = arrayType.getElementType();
      auto ctx = arrayType.getContext();
      
      // Map LLVM element types to TOSA supported types
      Type tensorElementType;
      if (auto intType = dyn_cast<IntegerType>(elementType)) {
        tensorElementType = intType;
      } else if (auto floatType = dyn_cast<FloatType>(elementType)) {
        tensorElementType = floatType;
      } else {
        // Default to f32 for unsupported types
        tensorElementType = FloatType::getF32(ctx);
      }
      
      return RankedTensorType::get({static_cast<int64_t>(arrayType.getNumElements())}, 
                                   tensorElementType);
    });
  }
};

// Convert LLVM constant operations to TOSA const
class LLVMConstantToTosaPattern : public ConversionPattern {
public:
  LLVMConstantToTosaPattern(TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter, LLVM::ConstantOp::getOperationName(),
                          1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto constOp = cast<LLVM::ConstantOp>(op);
    auto attr = constOp.getValue();
    
    // Convert scalar constants to tensor constants
    Type newType = getTypeConverter()->convertType(constOp.getType());
    if (!newType) {
      return failure();
    }
    
    // Create TOSA constant
    if (auto tensorType = dyn_cast<RankedTensorType>(newType)) {
      DenseElementsAttr denseAttr;
      if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
        denseAttr = DenseElementsAttr::get(tensorType, intAttr.getValue());
      } else if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
        denseAttr = DenseElementsAttr::get(tensorType, floatAttr.getValue());
      } else {
        return failure();
      }
      
      rewriter.replaceOpWithNewOp<tosa::ConstOp>(op, tensorType, denseAttr);
      return success();
    }
    
    return failure();
  }
};

// Convert LLVM add operation to TOSA add
class LLVMAddToTosaPattern : public ConversionPattern {
public:
  LLVMAddToTosaPattern(TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter, LLVM::AddOp::getOperationName(),
                          1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto addOp = cast<LLVM::AddOp>(op);
    
    // Convert result type
    Type newType = getTypeConverter()->convertType(addOp.getType());
    if (!newType) {
      return failure();
    }
    
    // Create TOSA add operation
    rewriter.replaceOpWithNewOp<tosa::AddOp>(op, newType, operands[0], operands[1]);
    return success();
  }
};

// Convert LLVM sub operation to TOSA sub
class LLVMSubToTosaPattern : public ConversionPattern {
public:
  LLVMSubToTosaPattern(TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter, LLVM::SubOp::getOperationName(),
                          1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto subOp = cast<LLVM::SubOp>(op);
    
    Type newType = getTypeConverter()->convertType(subOp.getType());
    if (!newType) {
      return failure();
    }
    
    rewriter.replaceOpWithNewOp<tosa::SubOp>(op, newType, operands[0], operands[1]);
    return success();
  }
};

// Convert LLVM mul operation to TOSA mul
class LLVMMulToTosaPattern : public ConversionPattern {
public:
  LLVMMulToTosaPattern(TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter, LLVM::MulOp::getOperationName(),
                          1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto mulOp = cast<LLVM::MulOp>(op);
    
    Type newType = getTypeConverter()->convertType(mulOp.getType());
    if (!newType) {
      return failure();
    }
    
    // TOSA mul requires a shift parameter for quantized operations
    // For now, use 0 shift for floating point operations
    auto shiftAttr = rewriter.getI8IntegerAttr(0);
    
    rewriter.replaceOpWithNewOp<tosa::MulOp>(op, newType, operands[0], operands[1], shiftAttr);
    return success();
  }
};

// Convert LLVM load operation to tensor extract
class LLVMLoadToTosaPattern : public ConversionPattern {
public:
  LLVMLoadToTosaPattern(TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter, LLVM::LoadOp::getOperationName(),
                          1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loadOp = cast<LLVM::LoadOp>(op);
    
    // For simplicity, assume loading from a tensor creates a scalar tensor
    Type elementType = loadOp.getType();
    auto ctx = elementType.getContext();
    auto tensorType = RankedTensorType::get({1}, elementType);
    
    // Create a slice operation to extract from the tensor
    // This is a simplified conversion - real implementation would need
    // proper address calculation
    auto startAttr = DenseI64ArrayAttr::get(ctx, {0});
    auto sizeAttr = DenseI64ArrayAttr::get(ctx, {1});
    
    rewriter.replaceOpWithNewOp<tosa::SliceOp>(op, tensorType, operands[0],
                                               startAttr, sizeAttr);
    return success();
  }
};

// Pass to convert LLVM IR to TOSA IR
class LLVMToTosaPass : public PassWrapper<LLVMToTosaPass, OperationPass<ModuleOp>> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tosa::TosaDialect, LLVM::LLVMDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();
    
    LLVMToTosaTypeConverter typeConverter;
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    
    // Set target dialect
    target.addLegalDialect<tosa::TosaDialect>();
    target.addIllegalDialect<LLVM::LLVMDialect>();
    
    // Add conversion patterns
    patterns.add<LLVMConstantToTosaPattern>(typeConverter, context);
    patterns.add<LLVMAddToTosaPattern>(typeConverter, context);
    patterns.add<LLVMSubToTosaPattern>(typeConverter, context);
    patterns.add<LLVMMulToTosaPattern>(typeConverter, context);
    patterns.add<LLVMLoadToTosaPattern>(typeConverter, context);
    
    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

// Create the LLVM to TOSA conversion pass
std::unique_ptr<Pass> createLLVMToTosaPass() {
  return std::make_unique<LLVMToTosaPass>();
}

// Command line tool
static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                 llvm::cl::desc("<input file>"),
                                                 llvm::cl::init("-"));

static llvm::cl::opt<std::string> outputFilename("o",
                                                  llvm::cl::desc("Output filename"),
                                                  llvm::cl::value_desc("filename"),
                                                  llvm::cl::init("-"));

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "LLVM IR to TOSA IR converter\n");

  MLIRContext context;
  context.getOrLoadDialect<LLVM::LLVMDialect>();
  context.getOrLoadDialect<tosa::TosaDialect>();
  context.getOrLoadDialect<func::FuncDialect>();

  // Read input file
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());

  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Failed to parse input file\n";
    return 1;
  }

  // Apply conversion pass
  PassManager pm(&context);
  pm.addPass(createLLVMToTosaPass());

  if (failed(pm.run(*module))) {
    llvm::errs() << "Conversion failed\n";
    return 1;
  }

  // Write output
  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  module->print(output->os());
  output->keep();

  return 0;
}