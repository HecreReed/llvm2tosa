#include "LLVMToTosaConverter.h"
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <llvm/IR/CFG.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/DominatorTree.h>

namespace llvm2tosa {

ControlFlowConverter::ControlFlowConverter(mlir::MLIRContext& context, TypeConverter& typeConverter)
    : context_(context), typeConverter_(typeConverter) {}

void ControlFlowConverter::convertControlFlow(llvm::Function& llvmFunc, mlir::func::FuncOp& tosaFunc) {
    // Analyze the control flow structure first
    analyzeControlFlow(llvmFunc);
    
    // Convert the function body
    mlir::Block* entryBlock = &tosaFunc.getBody().front();
    mlir::OpBuilder builder(entryBlock, entryBlock->end());
    
    // Process basic blocks in a structured manner
    for (auto& bb : llvmFunc) {
        convertBasicBlock(bb, builder);
    }
}

void ControlFlowConverter::analyzeControlFlow(llvm::Function& llvmFunc) {
    // Clear previous analysis
    loops_.clear();
    conditionals_.clear();
    
    // Identify loops and conditional structures
    identifyLoops(llvmFunc);
    identifyConditionals(llvmFunc);
}

void ControlFlowConverter::identifyLoops(llvm::Function& llvmFunc) {
    // Simple loop detection based on back edges
    // In a complete implementation, this would use LLVM's LoopInfo analysis
    
    for (auto& bb : llvmFunc) {
        // Check if this block has a back edge (successor that dominates it)
        for (auto* successor : successors(&bb)) {
            // Simplified check: if successor has lower address, it might be a loop header
            if (successor < &bb) {
                LoopInfo loop;
                loop.header = successor;
                loop.latch = &bb;
                loop.blocks.push_back(successor);
                loop.blocks.push_back(&bb);
                loop.isNaturalLoop = isNaturalLoop(loop.blocks);
                loops_.push_back(loop);
            }
        }
    }
}

void ControlFlowConverter::identifyConditionals(llvm::Function& llvmFunc) {
    // Identify if-then-else patterns
    for (auto& bb : llvmFunc) {
        auto* terminator = bb.getTerminator();
        
        if (auto* brInst = llvm::dyn_cast<llvm::BranchInst>(terminator)) {
            if (brInst->isConditional()) {
                IfInfo conditional;
                conditional.condition = &bb;
                conditional.thenBlock = brInst->getSuccessor(0);
                conditional.elseBlock = brInst->getSuccessor(1);
                
                // Try to find merge block (common successor)
                conditional.merge = nullptr;
                for (auto* thenSucc : successors(conditional.thenBlock)) {
                    for (auto* elseSucc : successors(conditional.elseBlock)) {
                        if (thenSucc == elseSucc) {
                            conditional.merge = thenSucc;
                            break;
                        }
                    }
                    if (conditional.merge) break;
                }
                
                conditionals_.push_back(conditional);
            }
        }
    }
}

void ControlFlowConverter::convertLoop(const LoopInfo& loop, mlir::OpBuilder& builder) {
    // Convert loop to TOSA while_loop operation
    
    // Create condition region
    auto whileOp = builder.create<mlir::tosa::WhileLoopOp>(
        builder.getUnknownLoc(),
        mlir::TypeRange{}, // result types
        mlir::ValueRange{} // initial values
    );
    
    // Build condition region
    mlir::Region& conditionRegion = whileOp.getCond();
    mlir::Block* condBlock = builder.createBlock(&conditionRegion);
    builder.setInsertionPointToEnd(condBlock);
    
    // Convert loop header to condition
    convertBasicBlock(*loop.header, builder);
    
    // For now, create a simple true condition
    auto boolType = mlir::RankedTensorType::get({1}, mlir::IntegerType::get(&context_, 1));
    mlir::Attribute trueAttr = mlir::DenseElementsAttr::get(boolType, 1);
    mlir::Value condition = builder.create<mlir::tosa::ConstOp>(
        builder.getUnknownLoc(), boolType, trueAttr);
    
    builder.create<mlir::tosa::YieldOp>(builder.getUnknownLoc(), condition);
    
    // Build body region
    mlir::Region& bodyRegion = whileOp.getBody();
    mlir::Block* bodyBlock = builder.createBlock(&bodyRegion);
    builder.setInsertionPointToEnd(bodyBlock);
    
    // Convert loop body blocks
    for (auto* block : loop.blocks) {
        if (block != loop.header) {
            convertBasicBlock(*block, builder);
        }
    }
    
    builder.create<mlir::tosa::YieldOp>(builder.getUnknownLoc(), mlir::ValueRange{});
}

void ControlFlowConverter::convertConditional(const IfInfo& conditional, mlir::OpBuilder& builder) {
    // Convert conditional to TOSA cond_if operation
    
    // Convert condition value
    mlir::Value conditionValue = convertCondition(
        conditional.condition->getTerminator()->getOperand(0), builder);
    
    auto ifOp = builder.create<mlir::tosa::CondIfOp>(
        builder.getUnknownLoc(),
        mlir::TypeRange{}, // result types
        conditionValue
    );
    
    // Build then region
    mlir::Region& thenRegion = ifOp.getThenBranch();
    mlir::Block* thenBlock = builder.createBlock(&thenRegion);
    builder.setInsertionPointToEnd(thenBlock);
    convertBasicBlock(*conditional.thenBlock, builder);
    builder.create<mlir::tosa::YieldOp>(builder.getUnknownLoc(), mlir::ValueRange{});
    
    // Build else region
    mlir::Region& elseRegion = ifOp.getElseBranch();
    mlir::Block* elseBlock = builder.createBlock(&elseRegion);
    builder.setInsertionPointToEnd(elseBlock);
    convertBasicBlock(*conditional.elseBlock, builder);
    builder.create<mlir::tosa::YieldOp>(builder.getUnknownLoc(), mlir::ValueRange{});
}

void ControlFlowConverter::convertBasicBlock(llvm::BasicBlock& bb, mlir::OpBuilder& builder) {
    // Convert all non-terminator instructions
    for (auto& inst : bb) {
        if (!inst.isTerminator()) {
            // This would be handled by InstructionConverter
            // For now, just skip non-terminator instructions
            continue;
        } else {
            // Handle terminator instructions
            convertTerminator(&inst, builder);
        }
    }
}

void ControlFlowConverter::convertTerminator(llvm::Instruction* terminator, mlir::OpBuilder& builder) {
    switch (terminator->getOpcode()) {
        case llvm::Instruction::Ret:
            convertReturn(llvm::cast<llvm::ReturnInst>(terminator), builder);
            break;
        case llvm::Instruction::Br:
            convertBranch(llvm::cast<llvm::BranchInst>(terminator), builder);
            break;
        case llvm::Instruction::Switch:
            convertSwitch(llvm::cast<llvm::SwitchInst>(terminator), builder);
            break;
        case llvm::Instruction::IndirectBr:
            convertIndirectBr(llvm::cast<llvm::IndirectBrInst>(terminator), builder);
            break;
        case llvm::Instruction::Invoke:
            convertInvoke(llvm::cast<llvm::InvokeInst>(terminator), builder);
            break;
        case llvm::Instruction::Unreachable:
            convertUnreachable(llvm::cast<llvm::UnreachableInst>(terminator), builder);
            break;
        default:
            // Unknown terminator
            break;
    }
}

void ControlFlowConverter::convertReturn(llvm::ReturnInst* ret, mlir::OpBuilder& builder) {
    if (ret->getReturnValue()) {
        // Return with value
        // The value should be converted by InstructionConverter
        // For now, create a placeholder
        auto returnType = typeConverter_.convertType(ret->getReturnValue()->getType());
        if (auto tensorType = returnType.dyn_cast<mlir::RankedTensorType>()) {
            mlir::Attribute zeroAttr = mlir::DenseElementsAttr::get(tensorType, 0);
            mlir::Value returnValue = builder.create<mlir::tosa::ConstOp>(
                builder.getUnknownLoc(), tensorType, zeroAttr);
            builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), returnValue);
        } else {
            builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
        }
    } else {
        // Void return
        builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
    }
}

void ControlFlowConverter::convertBranch(llvm::BranchInst* br, mlir::OpBuilder& builder) {
    if (br->isConditional()) {
        // Conditional branch - this should be handled by convertConditional
        // For now, just create a placeholder
        auto boolType = mlir::RankedTensorType::get({1}, mlir::IntegerType::get(&context_, 1));
        mlir::Attribute trueAttr = mlir::DenseElementsAttr::get(boolType, 1);
        mlir::Value condition = builder.create<mlir::tosa::ConstOp>(
            builder.getUnknownLoc(), boolType, trueAttr);
        
        // This would typically be converted to tosa.cond_if
    } else {
        // Unconditional branch - just continue to next block
        // In structured control flow, this is often just fall-through
    }
}

void ControlFlowConverter::convertSwitch(llvm::SwitchInst* sw, mlir::OpBuilder& builder) {
    // Convert switch to series of conditional operations
    // This is a complex transformation that would create nested cond_if operations
    
    mlir::Value switchValue = convertCondition(sw->getCondition(), builder);
    
    // For now, create a simple conditional based on the first case
    if (sw->getNumCases() > 0) {
        auto firstCase = sw->case_begin();
        mlir::Value caseValue = convertCondition(firstCase->getCaseValue(), builder);
        
        // Create comparison
        auto boolType = mlir::RankedTensorType::get({1}, mlir::IntegerType::get(&context_, 1));
        mlir::Value isEqual = builder.create<mlir::tosa::EqualOp>(
            builder.getUnknownLoc(), boolType, switchValue, caseValue);
        
        // Create conditional based on this comparison
        auto ifOp = builder.create<mlir::tosa::CondIfOp>(
            builder.getUnknownLoc(), mlir::TypeRange{}, isEqual);
        
        // Build case region
        mlir::Region& caseRegion = ifOp.getThenBranch();
        mlir::Block* caseBlock = builder.createBlock(&caseRegion);
        builder.setInsertionPointToEnd(caseBlock);
        builder.create<mlir::tosa::YieldOp>(builder.getUnknownLoc(), mlir::ValueRange{});
        
        // Build default region
        mlir::Region& defaultRegion = ifOp.getElseBranch();
        mlir::Block* defaultBlock = builder.createBlock(&defaultRegion);
        builder.setInsertionPointToEnd(defaultBlock);
        builder.create<mlir::tosa::YieldOp>(builder.getUnknownLoc(), mlir::ValueRange{});
    }
}

void ControlFlowConverter::convertIndirectBr(llvm::IndirectBrInst* ibr, mlir::OpBuilder& builder) {
    // Indirect branch is very difficult to convert to structured control flow
    // For now, just create a placeholder
    auto boolType = mlir::RankedTensorType::get({1}, mlir::IntegerType::get(&context_, 1));
    mlir::Attribute trueAttr = mlir::DenseElementsAttr::get(boolType, 1);
    mlir::Value condition = builder.create<mlir::tosa::ConstOp>(
        builder.getUnknownLoc(), boolType, trueAttr);
}

void ControlFlowConverter::convertInvoke(llvm::InvokeInst* invoke, mlir::OpBuilder& builder) {
    // Invoke instruction (exception handling) - convert to regular call for now
    // Exception handling would need additional infrastructure
    
    // Convert to function call
    // This would typically create a func.call operation
}

void ControlFlowConverter::convertUnreachable(llvm::UnreachableInst* unreachable, mlir::OpBuilder& builder) {
    // Unreachable instruction - in TOSA, this might be represented as an assertion failure
    // For now, just return from the function
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
}

mlir::Value ControlFlowConverter::convertCondition(llvm::Value* condition, mlir::OpBuilder& builder) {
    // Convert LLVM condition value to TOSA boolean tensor
    if (auto constInt = llvm::dyn_cast<llvm::ConstantInt>(condition)) {
        auto boolType = mlir::RankedTensorType::get({1}, mlir::IntegerType::get(&context_, 1));
        mlir::Attribute boolAttr = mlir::DenseElementsAttr::get(boolType, 
                                                               constInt->isOne() ? 1 : 0);
        return builder.create<mlir::tosa::ConstOp>(builder.getUnknownLoc(), boolType, boolAttr);
    }
    
    // For non-constant conditions, they should already be converted by InstructionConverter
    // For now, create a true condition as placeholder
    auto boolType = mlir::RankedTensorType::get({1}, mlir::IntegerType::get(&context_, 1));
    mlir::Attribute trueAttr = mlir::DenseElementsAttr::get(boolType, 1);
    return builder.create<mlir::tosa::ConstOp>(builder.getUnknownLoc(), boolType, trueAttr);
}

bool ControlFlowConverter::isNaturalLoop(const std::vector<llvm::BasicBlock*>& blocks) {
    // Simplified check for natural loop
    // A complete implementation would use proper dominance analysis
    return blocks.size() >= 2;
}

llvm::BasicBlock* ControlFlowConverter::findLoopLatch(llvm::BasicBlock* header) {
    // Find the block that branches back to the header
    for (auto* pred : predecessors(header)) {
        for (auto* succ : successors(pred)) {
            if (succ == header) {
                return pred;
            }
        }
    }
    return nullptr;
}

} // namespace llvm2tosa