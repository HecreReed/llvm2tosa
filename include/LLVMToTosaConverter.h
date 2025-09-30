#pragma once

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <iostream>
#include <sstream>

namespace llvm2tosa {

// Complete LLVM IR instruction set enumeration (68 instructions)
enum class LLVMOpcode {
    // Terminator Instructions (11)
    Ret, Br, Switch, IndirectBr, Invoke, Resume, Unreachable,
    CleanupRet, CatchRet, CatchSwitch, CallBr,
    
    // Unary Instructions (1)
    FNeg,
    
    // Binary Instructions (18)
    Add, FAdd, Sub, FSub, Mul, FMul, UDiv, SDiv, FDiv,
    URem, SRem, FRem, Shl, LShr, AShr, And, Or, Xor,
    
    // Memory Instructions (7)
    Alloca, Load, Store, GetElementPtr, Fence, AtomicCmpXchg, AtomicRMW,
    
    // Cast Instructions (14)
    Trunc, ZExt, SExt, FPToUI, FPToSI, UIToFP, SIToFP,
    FPTrunc, FPExt, PtrToInt, IntToPtr, BitCast, AddrSpaceCast, PtrToAddr,
    
    // Exception Handling (2)
    CleanupPad, CatchPad,
    
    // Other Instructions (15)
    ICmp, FCmp, PHI, Call, Select, UserOp1, UserOp2, VAArg,
    ExtractElement, InsertElement, ShuffleVector,
    ExtractValue, InsertValue, LandingPad, Freeze
};

// Complete TOSA operation set enumeration (66 operations)
enum class TOSAOpcode {
    // Tensor Operators (10)
    ArgMax, AvgPool2d, Conv2d, Conv3d, DepthwiseConv2d,
    FFT2d, MatMul, MaxPool2d, RFFT2d, TransposeConv2d,
    
    // Activation Functions (4)
    Clamp, Erf, Sigmoid, Tanh,
    
    // Elementwise Binary (18)
    Add, ArithmeticRightShift, BitwiseAnd, BitwiseOr, BitwiseXor,
    IntDiv, LogicalAnd, LogicalLeftShift, LogicalRightShift,
    LogicalOr, LogicalXor, Maximum, Minimum, Mul, Pow, Sub, Table,
    
    // Elementwise Unary (13)
    Abs, BitwiseNot, Ceil, Clz, Cos, Exp, Floor, Log,
    LogicalNot, Negate, Reciprocal, Rsqrt, Sin,
    
    // Selection (1)
    Select,
    
    // Comparison (3)
    Equal, Greater, GreaterEqual,
    
    // Reduction (6)
    ReduceAll, ReduceAny, ReduceMax, ReduceMin, ReduceProduct, ReduceSum,
    
    // Data Layout (8)
    Concat, Pad, Reshape, Reverse, Slice, Tile, Transpose,
    
    // Scatter/Gather (2)
    Gather, Scatter,
    
    // Image Operations (1)
    Resize,
    
    // Type Conversion (2)
    Cast, Rescale,
    
    // Data Nodes (2)
    Const, Identity,
    
    // Custom (1)
    Custom,
    
    // Control Flow (2)
    CondIf, WhileLoop,
    
    // Utility Operations (5)
    ApplyScale, Yield, Variable, VariableRead, VariableWrite,
    
    // Shape Operations (1)
    ConstShape
};

// Data types for conversion
enum class DataType {
    BOOL, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64,
    FLOAT16, BFLOAT16, FLOAT32, FLOAT64
};

// Tensor shape representation
struct TensorShape {
    std::vector<int64_t> dimensions;
    bool isDynamic = false;
    
    TensorShape() = default;
    TensorShape(std::vector<int64_t> dims) : dimensions(std::move(dims)) {}
    TensorShape(std::initializer_list<int64_t> dims) : dimensions(dims) {}
};

// Tensor type representation
struct TensorType {
    TensorShape shape;
    DataType elementType;
    
    TensorType() : elementType(DataType::FLOAT32) {}
    TensorType(TensorShape s, DataType t) : shape(std::move(s)), elementType(t) {}
};

// Value representation for SSA
struct Value {
    std::string name;
    TensorType type;
    bool isConstant = false;
    std::string constantValue;
    
    Value() = default;
    Value(const std::string& n, TensorType t) : name(n), type(std::move(t)) {}
};

// LLVM Basic Block representation
struct BasicBlock {
    std::string name;
    std::vector<std::string> instructions;
    std::vector<std::string> predecessors;
    std::vector<std::string> successors;
    bool isLoopHeader = false;
    bool isLoopLatch = false;
};

// Control flow analysis structures
struct LoopInfo {
    std::string header;
    std::string latch;
    std::vector<std::string> blocks;
    std::string preheader;
    std::vector<std::string> exits;
    bool isNaturalLoop = false;
    int depth = 0;
};

struct ConditionalInfo {
    std::string conditionBlock;
    std::string trueBlock;
    std::string falseBlock;
    std::string mergeBlock;
    std::string conditionValue;
};

// Memory allocation tracking
struct MemoryAllocation {
    TensorType tensorType;
    std::string llvmType;
    bool isGlobal = false;
    bool isArray = false;
    int64_t arraySize = 1;
    std::string initialValue;
};

// High-level pattern recognition structures
struct MatrixOperation {
    enum Type { 
        MATRIX_VECTOR_ADD, 
        MATRIX_MATRIX_ADD, 
        VECTOR_SCALAR_MUL,
        VECTOR_VECTOR_ADD,
        AXPY_OPERATION,     // a*x + y pattern
        DOT_PRODUCT,        // sum(a[i] * b[i]) pattern
        ELEMENT_WISE_OP, 
        UNKNOWN 
    };
    Type type;
    std::vector<std::string> inputTensors;
    std::vector<TensorShape> inputShapes;
    TensorShape outputShape;
    std::string operation; // "add", "mul", "axpy", etc.
    bool hasBroadcast = false;
    bool isDynamic = false; // for dynamic sized vectors
};

struct NestedLoopPattern {
    std::vector<std::string> inductionVars; // [i] for 1D, [i, j] for 2D loops
    std::vector<int64_t> bounds; // [n] for 1D, [rows, cols] for 2D
    std::vector<std::string> loopBlocks;
    std::string bodyBlock;
    bool isMatrixLoop = false;
    bool isVectorLoop = false;  // for 1D vector operations
    bool isDynamicSize = false; // when size comes from parameter
    MatrixOperation detectedOp;
};

struct FunctionSignature {
    std::string name;
    std::vector<std::pair<std::string, TensorType>> parameters;
    TensorType returnType;
    bool isVoidReturn = true;
};

/**
 * Complete LLVM IR to TOSA IR Converter
 * Handles all 68 LLVM instructions and converts to appropriate TOSA operations
 */
class LLVMToTosaConverter {
public:
    LLVMToTosaConverter();
    ~LLVMToTosaConverter() = default;

    // Main conversion interface
    std::string convertLLVMIRFile(const std::string& llvmIRCode);
    std::string convertLLVMIRToTOSA(const std::string& llvmIRCode);
    
    // Configuration
    void setOptimizationLevel(int level) { optimizationLevel_ = level; }
    void setDebugMode(bool enable) { debugMode_ = enable; }
    void setQuantizationMode(bool enable) { quantizationMode_ = enable; }
    bool getDebugMode() const { return debugMode_; }
    
private:
    // Core conversion methods
    void parseModule(const std::string& llvmIR);
    void convertGlobals();
    void convertFunctions();
    void convertBasicBlocks();
    void convertInstructions();
    
    // Helper methods for parsing
    size_t parseFunctionBody(const std::vector<std::string>& lines, 
                            size_t startIdx, const std::string& functionName);
    std::string convertGlobalVariable(const std::string& global);
    std::string parseConstantValue(const std::string& value, const TensorType& type);
    std::string convertInstruction(LLVMOpcode opcode, const std::string& instruction);
    
    // Instruction conversion methods for all 68 LLVM instructions
    std::string convertTerminatorInstruction(LLVMOpcode opcode, const std::string& instruction);
    std::string convertUnaryInstruction(LLVMOpcode opcode, const std::string& instruction);
    std::string convertBinaryInstruction(LLVMOpcode opcode, const std::string& instruction);
    std::string convertMemoryInstruction(LLVMOpcode opcode, const std::string& instruction);
    std::string convertCastInstruction(LLVMOpcode opcode, const std::string& instruction);
    std::string convertComparisonInstruction(LLVMOpcode opcode, const std::string& instruction);
    std::string convertVectorInstruction(LLVMOpcode opcode, const std::string& instruction);
    std::string convertAggregateInstruction(LLVMOpcode opcode, const std::string& instruction);
    std::string convertExceptionInstruction(LLVMOpcode opcode, const std::string& instruction);
    std::string convertOtherInstruction(LLVMOpcode opcode, const std::string& instruction);
    
    // Type system conversion
    TensorType convertLLVMTypeToTensorType(const std::string& llvmType);
    DataType convertLLVMTypeToDataType(const std::string& llvmType);
    TensorShape inferTensorShape(const std::string& llvmType);
    
    // Memory model abstraction
    std::string convertAllocaToTensorInit(const std::string& instruction);
    std::string convertLoadToTensorSlice(const std::string& instruction);
    std::string convertStoreToTensorUpdate(const std::string& instruction);
    std::string convertGEPToTensorIndex(const std::string& instruction);
    
    // Control flow conversion
    void analyzeControlFlow();
    void identifyLoops();
    void identifyConditionals();
    std::string convertLoopToWhileLoop(const LoopInfo& loop);
    std::string convertConditionalToCondIf(const ConditionalInfo& conditional);
    
    // High-level pattern recognition and analysis
    void analyzeHighLevelPatterns();
    NestedLoopPattern analyzeNestedLoops(const std::string& functionName);
    MatrixOperation detectMatrixOperation(const NestedLoopPattern& loopPattern);
    MatrixOperation detectVectorOperation(const NestedLoopPattern& loopPattern);
    MatrixOperation detectAXPYPattern(const NestedLoopPattern& loopPattern);
    MatrixOperation detectDotProductPattern(const NestedLoopPattern& loopPattern);
    FunctionSignature inferTensorSignature(const std::string& functionName);
    bool isMatrixVectorBroadcast(const NestedLoopPattern& pattern);
    bool isVectorOperation(const NestedLoopPattern& pattern);
    TensorShape inferShapeFromLoopBounds(const std::vector<int64_t>& bounds);
    TensorShape inferDynamicVectorShape();
    std::string generateHighLevelTOSA(const MatrixOperation& matrixOp, const std::string& funcName);
    
    // TOSA operation generation
    std::string generateTOSAOperation(TOSAOpcode opcode, 
                                     const std::vector<std::string>& inputs,
                                     const std::vector<TensorType>& inputTypes,
                                     const TensorType& outputType,
                                     const std::map<std::string, std::string>& attributes = {});
    
    // Tensor operations
    std::string createTensorFromScalar(const std::string& scalarValue, DataType type);
    std::string broadcastTensors(const std::string& lhs, const std::string& rhs,
                                const TensorType& lhsType, const TensorType& rhsType);
    std::string reshapeTensor(const std::string& tensor, const TensorShape& newShape);
    std::string ensureTensorValue(const std::string& value, const TensorType& expectedType);
    
    // Utility methods
    std::string generateUniqueName(const std::string& prefix = "tmp");
    LLVMOpcode parseInstructionOpcode(const std::string& instruction);
    std::vector<std::string> parseOperands(const std::string& instruction);
    std::string parseResultName(const std::string& instruction);
    std::string parseResultType(const std::string& instruction);
    
    // Output generation
    std::string generateTOSAModule();
    std::string generateTOSAFunction(const std::string& functionName);
    
    // State management
    std::map<std::string, Value> valueMapping_;
    std::map<std::string, BasicBlock> basicBlocks_;
    std::map<std::string, MemoryAllocation> memoryAllocations_;
    std::vector<LoopInfo> loops_;
    std::vector<ConditionalInfo> conditionals_;
    std::vector<std::string> globalVariables_;
    std::vector<std::string> functions_;
    
    // High-level pattern recognition state
    std::map<std::string, NestedLoopPattern> detectedLoopPatterns_;
    std::map<std::string, MatrixOperation> detectedMatrixOps_;
    std::map<std::string, FunctionSignature> tensorSignatures_;
    
    // Generated TOSA code
    std::stringstream tosaOutput_;
    
    // Configuration
    int optimizationLevel_ = 0;
    bool debugMode_ = false;
    bool quantizationMode_ = false;
    
    // Counters
    int uniqueCounter_ = 0;
    
    // Current context
    std::string currentFunction_;
    std::string currentBlock_;
};

// Utility functions for LLVM IR parsing
namespace utils {
    std::vector<std::string> splitLines(const std::string& text);
    std::string trim(const std::string& str);
    bool isInstruction(const std::string& line);
    bool isTerminator(const std::string& line);
    std::string extractFunctionName(const std::string& line);
    std::string extractBasicBlockName(const std::string& line);
    std::vector<std::string> parseArguments(const std::string& args);
    std::string formatTensorType(const TensorType& type);
    std::string escapeStringForMLIR(const std::string& str);
}

} // namespace llvm2tosa