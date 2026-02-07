"""

main.py
Agentic Kernel Optimizer (AKO) demo using TileLang, verl RL agent, AITER, and LangChain orchestration.
"""

import sys
import os

# Add the parent directory of AKO_Project to the Python path
# This allows importing modules within AKO_Project correctly.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.kernel_generator import KernelGenerator
from src.profiling_feedback import ProfilingFeedback
from src.hardware_interface import HardwareInterface
from src.agentic_loop import AgenticLoop

def main():

    print("Initializing AKO agentic components...")
    hardware_interface = HardwareInterface(target_hardware="MI300X")
    kernel_generator = KernelGenerator()  # Now uses LangChain and TileLang
    profiling_feedback = ProfilingFeedback()
    agentic_loop = AgenticLoop(hardware_interface, kernel_generator, profiling_feedback)

    high_level_kernel_spec = "Efficiently multiply two large matrices (GEMM) for AI inference."
    max_optimization_iterations = 5

    print(f"\n--- Starting AKO Agentic Optimization for: {high_level_kernel_spec} ---")
    best_kernel, best_perf = agentic_loop.run_loop(high_level_kernel_spec, max_optimization_iterations)
    print(f"\n--- AKO Agentic Optimization Finished ---")

    if best_kernel:
        print(f"\nAchieved best performance: {best_perf:.2f} ms")
        print("\nFinal Optimized Kernel Code:")
        print(best_kernel)
    else:
        print("\nOptimization did not yield a valid kernel.")

    # Ensure all temporary files are cleaned up at the end of the run
    hardware_interface._cleanup_temp_files()

if __name__ == "__main__":
    main()
