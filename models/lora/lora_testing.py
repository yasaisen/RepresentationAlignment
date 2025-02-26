"""
 Copyright (c) 2025, yasaisen.
 All rights reserved.

 last modified in 2502141330
"""

import torch
import torch.nn as nn
from typing import Dict
import numpy as np
from typing import Tuple, Dict, Any
import os
from lora import *

class SimpleTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 5)
        
    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x

def verify_weights_unchanged(
    original_state: Dict[str, torch.Tensor],
    model: nn.Module
) -> Tuple[bool, float]:
    """驗證模型權重是否保持不變"""
    current_state = model.state_dict()
    max_diff = 0.0
    
    for key in original_state:
        if key in current_state:
            diff = (original_state[key] - current_state[key]).abs().max().item()
            max_diff = max(max_diff, diff)
    
    return max_diff < 1e-6, max_diff

def test_lora_initialization(
    batch_size: int = 32,
    seed: int = 42,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    rank: int = 8,
    alpha: float = 1.0
) -> Tuple[bool, Dict[str, Any]]:
    """
    測試LoRA初始化後的輸出是否與原始模型一致
    """
    # 設置隨機種子
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 創建測試結果字典
    test_results = {
        "input_shape": None,
        "original_output_shape": None,
        "lora_output_shape": None,
        "max_abs_diff": None,
        "max_rel_diff": None,
        "weights_unchanged": False,
        "max_weight_diff": None,
        "within_tolerance": False,
        "shapes_match": False
    }
    
    # 創建測試模型和輸入
    base_model = SimpleTestModel()
    original_state = {k: v.clone() for k, v in base_model.state_dict().items()}
    x = torch.randn(batch_size, 10)
    test_results["input_shape"] = list(x.shape)
    
    # 獲取原始模型輸出
    base_model.eval()
    with torch.no_grad():
        original_output = base_model(x)
    test_results["original_output_shape"] = list(original_output.shape)
    
    # 創建LoRA包裝模型
    model_with_lora = ModelWithLoRA(base_model)
    
    # 添加LoRA
    model_with_lora.add_lora(rank=rank, alpha=alpha, dropout=0.0)
    
    # 檢查原始權重是否保持不變
    weights_unchanged, max_weight_diff = verify_weights_unchanged(
        original_state,
        model_with_lora.base_model
    )
    test_results["weights_unchanged"] = weights_unchanged
    test_results["max_weight_diff"] = float(max_weight_diff)
    
    # 獲取LoRA模型輸出
    model_with_lora.base_model.eval()
    with torch.no_grad():
        lora_output = model_with_lora.base_model(x)
    test_results["lora_output_shape"] = list(lora_output.shape)
    
    # 計算差異
    abs_diff = torch.abs(original_output - lora_output)
    rel_diff = abs_diff / (torch.abs(original_output) + 1e-9)
    
    test_results["max_abs_diff"] = float(torch.max(abs_diff))
    test_results["max_rel_diff"] = float(torch.max(rel_diff))
    
    # 檢查形狀是否匹配
    shapes_match = (original_output.shape == lora_output.shape)
    test_results["shapes_match"] = shapes_match
    
    # 檢查數值是否在容差範圍內
    within_tolerance = torch.allclose(
        original_output, 
        lora_output,
        rtol=rtol,
        atol=atol
    )
    test_results["within_tolerance"] = within_tolerance
    
    # 恢復原始權重
    model_with_lora.restore_original_weights()
    
    # 驗證恢復後的權重
    restored_unchanged, restored_max_diff = verify_weights_unchanged(
        original_state,
        model_with_lora.base_model
    )
    test_results["weights_restored"] = restored_unchanged
    test_results["restored_max_diff"] = float(restored_max_diff)
    
    all_passed = (
        shapes_match and 
        within_tolerance and 
        weights_unchanged and 
        restored_unchanged
    )
    
    return all_passed, test_results

def test_lora_merge_unmerge(
    batch_size: int = 32,
    seed: int = 42,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    rank: int = 8,
    alpha: float = 1.0
) -> Tuple[bool, Dict[str, Any]]:
    """
    測試LoRA權重合併與解除合併的一致性
    """
    # 設置隨機種子
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    test_results = {
        "input_shape": None,
        "original_output_shape": None,
        "unmerged_output_shape": None,
        "merged_output_shape": None,
        "after_unmerge_output_shape": None,
        "max_diff_merged_unmerged": None,
        "max_diff_after_unmerge": None,
        "shapes_consistent": False,
        "outputs_match": False,
        "weights_properly_restored": False
    }
    
    # 創建測試數據
    x = torch.randn(batch_size, 10)
    test_results["input_shape"] = list(x.shape)
    
    # 創建模型和LoRA包裝
    base_model = SimpleTestModel()
    original_state = {k: v.clone() for k, v in base_model.state_dict().items()}
    model_with_lora = ModelWithLoRA(base_model)
    
    # 添加LoRA並初始化非零權重
    model_with_lora.add_lora(rank=rank, alpha=alpha, dropout=0.0)
    
    # 初始化非零的LoRA權重
    for name, layer in model_with_lora.lora_layers.items():
        # 使用一些非零值初始化LoRA矩陣
        torch.nn.init.normal_(layer.lora_A, mean=0.0, std=0.02)
        torch.nn.init.normal_(layer.lora_B, mean=0.0, std=0.02)
    
    # 保存未合併時的權重
    unmerged_weights = {k: v.clone() for k, v in model_with_lora.base_model.state_dict().items()}
    
    # 測試步驟1: 未合併狀態下的輸出
    model_with_lora.base_model.eval()
    with torch.no_grad():
        unmerged_output = model_with_lora.base_model(x)
    test_results["unmerged_output_shape"] = list(unmerged_output.shape)
    
    # 測試步驟2: 合併權重後的輸出
    for layer in model_with_lora.lora_layers.values():
        layer.merge_weights()
    
    merged_weights = {k: v.clone() for k, v in model_with_lora.base_model.state_dict().items()}
    
    with torch.no_grad():
        merged_output = model_with_lora.base_model(x)
    test_results["merged_output_shape"] = list(merged_output.shape)
    
    # 測試步驟3: 解除合併後的輸出
    for layer in model_with_lora.lora_layers.values():
        layer.unmerge_weights()
    
    with torch.no_grad():
        after_unmerge_output = model_with_lora.base_model(x)
    test_results["after_unmerge_output_shape"] = list(after_unmerge_output.shape)
    
    # 計算差異
    diff_merged_unmerged = torch.abs(merged_output - unmerged_output)
    diff_after_unmerge = torch.abs(after_unmerge_output - unmerged_output)
    
    test_results["max_diff_merged_unmerged"] = float(torch.max(diff_merged_unmerged))
    test_results["max_diff_after_unmerge"] = float(torch.max(diff_after_unmerge))
    
    # 驗證形狀一致性
    shapes_match = (
        unmerged_output.shape == merged_output.shape == after_unmerge_output.shape
    )
    test_results["shapes_consistent"] = shapes_match
    
    # 驗證輸出一致性
    outputs_match = (
        torch.allclose(merged_output, unmerged_output, rtol=rtol, atol=atol) and
        torch.allclose(after_unmerge_output, unmerged_output, rtol=rtol, atol=atol)
    )
    test_results["outputs_match"] = outputs_match
    
    # 驗證權重恢復
    unmerged_state = model_with_lora.base_model.state_dict()
    weights_restored = all(
        torch.allclose(unmerged_state[key], unmerged_weights[key], rtol=rtol, atol=atol)
        for key in unmerged_state
    )
    test_results["weights_properly_restored"] = weights_restored
    
    # 添加權重差異的詳細信息
    test_results["weight_changes"] = {
        "max_merge_diff": float(max(
            torch.max(abs(merged_weights[k] - unmerged_weights[k]))
            for k in merged_weights
        )),
        "max_unmerge_diff": float(max(
            torch.max(abs(unmerged_state[k] - unmerged_weights[k]))
            for k in unmerged_state
        ))
    }
    
    all_passed = shapes_match and outputs_match and weights_restored
    
    return all_passed, test_results

def test_lora_save_load(
    batch_size: int = 32,
    seed: int = 42,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    rank: int = 8,
    alpha: float = 1.0,
    save_dir: str = "./lora_test_weights"
) -> Tuple[bool, Dict[str, Any]]:
    """
    測試LoRA權重的儲存和載入
    包含非零權重情況下的一致性檢查
    """
    # 設置隨機種子
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 創建測試結果字典
    test_results = {
        "pre_save_output_shape": None,
        "post_load_output_shape": None,
        "max_weight_diff": None,
        "max_output_diff": None,
        "shapes_match": False,
        "weights_match": False,
        "outputs_match": False,
        "save_load_successful": False
    }
    
    # 創建保存目錄
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "lora_weights.pth")
    
    try:
        # 創建基礎模型和測試輸入
        x = torch.randn(batch_size, 10)

        # 創建並初始化第一個LoRA模型
        base_model1 = SimpleTestModel()
        base_model_state = base_model1.state_dict()
        model_with_lora1 = ModelWithLoRA(base_model1)
        model_with_lora1.add_lora(rank=rank, alpha=alpha, dropout=0.0)
        
        # 模擬訓練：手動設置一些非零的LoRA權重
        for name, layer in model_with_lora1.lora_layers.items():
            # 設置一些非零值
            with torch.no_grad():
                layer.lora_A.data = torch.randn_like(layer.lora_A) * 0.1
                layer.lora_B.data = torch.randn_like(layer.lora_B) * 0.1
        
        # 獲取第一個模型的輸出
        model_with_lora1.base_model.eval()
        with torch.no_grad():
            output1 = model_with_lora1.base_model(x)
        test_results["pre_save_output_shape"] = list(output1.shape)
        
        # 儲存LoRA權重
        lora_save(model_with_lora1, save_path, False)
        
        # 創建新的模型並載入權重
        base_model2 = SimpleTestModel()
        base_model2.load_state_dict(base_model_state)
        model_with_lora2 = ModelWithLoRA(base_model2)
        model_with_lora2.add_lora(rank=rank, alpha=alpha, dropout=0.0)
        
        # 載入儲存的權重
        lora_load(model_with_lora2, save_path)
        
        # 獲取載入後模型的輸出
        model_with_lora2.base_model.eval()
        with torch.no_grad():
            output2 = model_with_lora2.base_model(x)
        test_results["post_load_output_shape"] = list(output2.shape)
        
        # 檢查權重是否匹配
        max_weight_diff = 0.0
        for name in model_with_lora1.lora_layers:
            layer1 = model_with_lora1.lora_layers[name]
            layer2 = model_with_lora2.lora_layers[name]
            
            a_diff = (layer1.lora_A - layer2.lora_A).abs().max().item()
            b_diff = (layer1.lora_B - layer2.lora_B).abs().max().item()
            max_weight_diff = max(max_weight_diff, a_diff, b_diff)
        
        test_results["max_weight_diff"] = float(max_weight_diff)
        test_results["weights_match"] = max_weight_diff < atol
        
        # 檢查輸出是否匹配
        output_diff = (output1 - output2).abs()
        max_output_diff = float(output_diff.max())
        test_results["max_output_diff"] = max_output_diff
        test_results["outputs_match"] = torch.allclose(output1, output2, rtol=rtol, atol=atol)
        
        # 檢查形狀是否匹配
        test_results["shapes_match"] = (output1.shape == output2.shape)
        
        # 標記儲存載入成功
        test_results["save_load_successful"] = True
        
    except Exception as e:
        print(f"Error during save/load test: {str(e)}")
        test_results["save_load_successful"] = False
    finally:
        # 清理儲存的檔案
        if os.path.exists(save_path):
            os.remove(save_path)
        
    # 判斷整體測試是否通過
    all_passed = (
        test_results["save_load_successful"] and
        test_results["shapes_match"] and
        test_results["weights_match"] and
        test_results["outputs_match"]
    )
    
    return all_passed, test_results

def print_test_results(test_results: Dict[str, Any]) -> None:
    """打印測試結果"""
    print("\n=== LoRA Initialization Test Results ===")

    print("\n--- Shapes ---")
    print(f"Input Shape: {test_results['input_shape']}")
    print(f"Original Output Shape: {test_results['original_output_shape']}")
    print(f"LoRA Output Shape: {test_results['lora_output_shape']}")
    print(f"Shapes Match: {test_results['shapes_match']}")

    print("\n--- Numerical Differences ---")
    print(f"Maximum Absolute Difference: {test_results['max_abs_diff']:.2e}")
    print(f"Maximum Relative Difference: {test_results['max_rel_diff']:.2e}")
    print(f"Within Tolerance: {test_results['within_tolerance']}")

    print("\n--- Weight Verification ---")
    print(f"Original Weights Unchanged: {test_results['weights_unchanged']}")
    print(f"Maximum Weight Difference: {test_results['max_weight_diff']:.2e}")
    print(f"Weights Properly Restored: {test_results['weights_restored']}")
    print(f"Restored Weight Max Difference: {test_results['restored_max_diff']:.2e}")
    print("=====================================")

def print_merge_test_results(test_results: Dict[str, Any]) -> None:
    """打印合併測試結果"""
    print("\n=== LoRA Merge/Unmerge Test Results ===")

    print("\n--- Shapes ---")
    print(f"Input Shape: {test_results['input_shape']}")
    print(f"Unmerged Output Shape: {test_results['unmerged_output_shape']}")
    print(f"Merged Output Shape: {test_results['merged_output_shape']}")
    print(f"After Unmerge Output Shape: {test_results['after_unmerge_output_shape']}")
    print(f"Shapes Consistent: {test_results['shapes_consistent']}")
    
    print("\n--- Output Differences ---")
    print(f"Max Difference (Merged vs Unmerged): {test_results['max_diff_merged_unmerged']:.2e}")
    print(f"Max Difference (After Unmerge vs Original): {test_results['max_diff_after_unmerge']:.2e}")
    print(f"Outputs Match: {test_results['outputs_match']}")
    
    print("\n--- Weight Changes ---")
    print(f"Max Weight Change During Merge: {test_results['weight_changes']['max_merge_diff']:.2e}")
    print(f"Max Weight Change After Unmerge: {test_results['weight_changes']['max_unmerge_diff']:.2e}")
    print(f"Weights Properly Restored: {test_results['weights_properly_restored']}")
    print("=====================================")

def print_save_load_test_results(test_results: Dict[str, Any]) -> None:
    """打印儲存載入測試結果"""
    print("\n=== LoRA Save/Load Test Results ===")

    print("\n--- Shapes ---")
    print(f"Pre-save Output Shape: {test_results['pre_save_output_shape']}")
    print(f"Post-load Output Shape: {test_results['post_load_output_shape']}")
    print(f"Shapes Match: {test_results['shapes_match']}")
    
    print("\n--- Weights ---")
    print(f"Maximum Weight Difference: {test_results['max_weight_diff']:.2e}")
    print(f"Weights Match: {test_results['weights_match']}")
    
    print("\n--- Outputs ---")
    print(f"Maximum Output Difference: {test_results['max_output_diff']:.2e}")
    print(f"Outputs Match: {test_results['outputs_match']}")
    
    print("\n--- Overall ---")
    print(f"Save/Load Operation Successful: {test_results['save_load_successful']}")
    print("=====================================")

def run_comprehensive_tests():
    """運行綜合測試"""
    print("Starting comprehensive LoRA tests...")
    
    # 基本測試
    print("\n=== Basic Test ===")
    passed, results = test_lora_initialization()
    print_test_results(results)
    print(f"Basic Test {'PASSED' if passed else 'FAILED'}\n")
    
    # # 不同批次大小的測試
    # batch_sizes = [1, 16, 64, 128]
    # for batch_size in batch_sizes:
    #     print(f"\n=== Batch Size {batch_size} Test ===")
    #     passed, results = test_lora_initialization(batch_size=batch_size)
    #     print_test_results(results)
    #     print(f"Batch Size {batch_size} Test {'PASSED' if passed else 'FAILED'}")
    
    # # 不同容差的測試
    # tolerances = [(1e-4, 1e-4), (1e-6, 1e-6), (1e-8, 1e-8)]
    # for rtol, atol in tolerances:
    #     print(f"\n=== Tolerance Test (rtol={rtol}, atol={atol}) ===")
    #     passed, results = test_lora_initialization(rtol=rtol, atol=atol)
    #     print_test_results(results)
    #     print(f"Tolerance Test {'PASSED' if passed else 'FAILED'}")
    
    # # 不同LoRA參數的測試
    # lora_configs = [
    #     (4, 0.5),
    #     (8, 1.0),
    #     (16, 2.0)
    # ]
    # for rank, alpha in lora_configs:
    #     print(f"\n=== LoRA Config Test (rank={rank}, alpha={alpha}) ===")
    #     passed, results = test_lora_initialization(rank=rank, alpha=alpha)
    #     print_test_results(results)
    #     print(f"LoRA Config Test {'PASSED' if passed else 'FAILED'}")

def run_merge_unmerge_tests():
    """運行合併/解除合併測試"""
    print("Starting LoRA merge/unmerge tests...")
    
    # 基本測試
    print("\n=== Basic Merge/Unmerge Test ===")
    passed, results = test_lora_merge_unmerge()
    print_merge_test_results(results)
    print(f"Basic Test {'PASSED' if passed else 'FAILED'}\n")
    
    # # 不同批次大小的測試
    # batch_sizes = [1, 16, 64, 128]
    # for batch_size in batch_sizes:
    #     print(f"\n=== Batch Size {batch_size} Merge/Unmerge Test ===")
    #     passed, results = test_lora_merge_unmerge(batch_size=batch_size)
    #     print_merge_test_results(results)
    #     print(f"Batch Size {batch_size} Test {'PASSED' if passed else 'FAILED'}")
    
    # # 不同LoRA參數的測試
    # lora_configs = [
    #     (4, 0.5),
    #     (8, 1.0),
    #     (16, 2.0)
    # ]
    # for rank, alpha in lora_configs:
    #     print(f"\n=== LoRA Config (rank={rank}, alpha={alpha}) Merge/Unmerge Test ===")
    #     passed, results = test_lora_merge_unmerge(rank=rank, alpha=alpha)
    #     print_merge_test_results(results)
    #     print(f"LoRA Config Test {'PASSED' if passed else 'FAILED'}")

def run_save_load_tests():
    """運行儲存載入測試"""
    print("Starting LoRA save/load tests...")
    
    # 基本測試
    print("\n=== Basic Save/Load Test ===")
    passed, results = test_lora_save_load()
    print_save_load_test_results(results)
    print(f"Basic Save/Load Test {'PASSED' if passed else 'FAILED'}")
    
    # # 不同LoRA配置的測試
    # configs = [
    #     (4, 0.5),
    #     (16, 2.0),
    #     (32, 4.0)
    # ]
    # for rank, alpha in configs:
    #     print(f"\n=== Save/Load Test (rank={rank}, alpha={alpha}) ===")
    #     passed, results = test_lora_save_load(rank=rank, alpha=alpha)
    #     print_save_load_test_results(results)
    #     print(f"Config Test {'PASSED' if passed else 'FAILED'}")
    
    # # 不同容差的測試
    # tolerances = [(1e-4, 1e-4), (1e-6, 1e-6), (1e-8, 1e-8)]
    # for rtol, atol in tolerances:
    #     print(f"\n=== Save/Load Test (rtol={rtol}, atol={atol}) ===")
    #     passed, results = test_lora_save_load(rtol=rtol, atol=atol)
    #     print_save_load_test_results(results)
    #     print(f"Tolerance Test {'PASSED' if passed else 'FAILED'}")



