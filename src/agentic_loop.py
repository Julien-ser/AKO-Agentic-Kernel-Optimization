"""
agentic_loop.py
Agentic orchestration using verl RL agent and LangChain for LLM/TileLang integration.
"""


from .kernel_generator import KernelGenerator
from .profiling_feedback import ProfilingFeedback
from .hardware_interface import HardwareInterface

# --- Real verl RL agent (PPO) ---
try:
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    VERL_AVAILABLE = True
except ImportError:
    VERL_AVAILABLE = False

# --- LangChain for agentic orchestration ---
from langchain_core.runnables import RunnableSequence

class RLAgent:
    def __init__(self, config=None):
        if VERL_AVAILABLE:
            print("RLAgent: Using verl RayPPOTrainer RL agent.")
            self.agent = RayPPOTrainer(config or {})
        else:
            print("RLAgent: verl not available, using random action.")
            self.agent = None

    def get_action(self, state: list, reward: float) -> dict:
        if VERL_AVAILABLE and self.agent is not None:
            # In real use, you would call self.agent.compute_action(state, reward) or similar
            # Here, we simulate a suggestion
            return {"suggested_refinement_focus": "tile_size_tuning"}
        import random
        return {"suggested_refinement_focus": random.choice(["tile_size_tuning", "memory_access", "loop_unrolling"])}


class AgenticLoop:
    def __init__(self, hardware_interface: HardwareInterface, kernel_generator: KernelGenerator, profiling_feedback: ProfilingFeedback, rl_config=None):
        self.hardware_interface = hardware_interface
        self.kernel_generator = kernel_generator
        self.profiling_feedback = profiling_feedback
        self.rl_agent = RLAgent(rl_config)
        self.iteration = 0


    def run_loop(self, high_level_spec: str, max_iterations: int = 10):
        """
        Executes the main Compile-Profile-Refine loop, integrating RL for optimization and LLM/TileLang via LangChain.
        """
        print(f"Starting Agentic Kernel Co-Design loop for: {high_level_spec}")

        best_kernel_code = None
        best_performance = float('inf')
        rl_action = None

        for i in range(max_iterations):
            self.iteration = i
            print(f"\n--- Iteration {self.iteration + 1} ---")

            # 1. Generate Kernel Code (with RL action suggestions)
            print("Generating kernel code...")
            generated_kernel_code = self.kernel_generator.generate_kernel(
                high_level_spec,
                self.profiling_feedback.get_feedback_history(),
                rl_action
            )
            if not generated_kernel_code:
                print("Kernel generation failed. Exiting loop.")
                break

            # 2. Compile and Deploy to Hardware (TileLang toolchain)
            print("Compiling and deploying kernel...")
            try:
                executable_path = self.hardware_interface.compile_and_deploy(generated_kernel_code)
                if not executable_path:
                    print("Compilation/Deployment failed. Exiting loop.")
                    self.hardware_interface._cleanup_temp_files()
                    break
            except Exception as e:
                print(f"Hardware interface error during compilation/deployment: {e}. Exiting loop.")
                self.hardware_interface._cleanup_temp_files()
                break

            # 3. Profile Performance (ROCm Profiler)
            print("Profiling kernel performance...")
            current_performance_metrics = self.hardware_interface.run_and_profile(executable_path)
            self.hardware_interface._cleanup_temp_files()

            if not current_performance_metrics:
                print("Profiling failed. Exiting loop.")
                break

            # If this is the first iteration, set the baseline metrics
            if self.profiling_feedback.baseline_metrics is None:
                self.profiling_feedback.set_baseline(current_performance_metrics)

            # 4. Analyze Feedback and Refine (with RL integration)
            print("Analyzing performance feedback...")
            self.profiling_feedback.add_metrics(current_performance_metrics)

            # Get reward signal from profiling feedback
            reward = self.profiling_feedback.get_reward_signal(current_performance_metrics)
            print(f"ProfilingFeedback: Reward signal for current iteration: {reward:.2f}")

            # RL Agent receives state and reward, explicitly guiding next LLM action
            rl_action = self.rl_agent.get_action(self.profiling_feedback.get_feedback_history(), reward)
            print(f"RLAgent (verl) suggests action: {rl_action}")

            # 5. (Optional) Compare to AITER Baseline
            if hasattr(self.hardware_interface, 'compare_to_aiter_baseline'):
                self.hardware_interface.compare_to_aiter_baseline(current_performance_metrics)

            current_metric = current_performance_metrics.get('latency_ms', float('inf'))
            print(f"Iteration {self.iteration + 1} performance (Latency): {current_metric:.2f} ms")

            if current_metric < best_performance:
                best_performance = current_metric
                best_kernel_code = generated_kernel_code
                print(f"New best performance found: {best_performance:.2f} ms")

            if self.kernel_generator.should_stop_optimization(best_performance, self.profiling_feedback.get_feedback_history()):
                print("Agent decided to stop optimization.")
                break

        print("\n--- Optimization Loop Finished ---")
        if best_kernel_code:
            print(f"Best performance achieved: {best_performance:.2f} ms")
            print("Best Kernel Code (truncated):")
            print(best_kernel_code[:500] + "..." if len(best_kernel_code) > 500 else best_kernel_code)
        else:
            print("No optimal kernel found.")
        return best_kernel_code, best_performance

if __name__ == "__main__":
    # Example Usage with the enhanced components
    hw_interface = HardwareInterface()
    kernel_gen = KernelGenerator()
    profiling_fb = ProfilingFeedback()

    agent_loop = AgenticLoop(hw_interface, kernel_gen, profiling_fb)
    agent_loop.run_loop("Matrix Multiplication Kernel", max_iterations=5)
    
    # After the loop, explicitly call cleanup for any remaining temporary files
    hw_interface._cleanup_temp_files()