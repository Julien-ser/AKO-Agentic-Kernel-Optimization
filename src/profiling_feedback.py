"""
profiling_feedback.py
Handles processing performance metrics from the ROCm Profiler (rocprof)
and providing feedback to the kernel generation agent.
"""

import math

class ProfilingFeedback:
    def __init__(self):
        self._feedback_history = []  # Stores historical performance metrics
        self.baseline_metrics = None # Stores initial/target performance metrics

    def set_baseline(self, metrics: dict):
        """
        Sets the baseline performance metrics against which future iterations will be compared.

        Args:
            metrics (dict): A dictionary of baseline performance metrics.
        """
        self.baseline_metrics = metrics
        print(f"ProfilingFeedback: Baseline metrics set to: {self.baseline_metrics}")

    def add_metrics(self, metrics: dict):
        """
        Adds a new set of performance metrics to the history.

        Args:
            metrics (dict): A dictionary of performance metrics (e.g., {'latency_ms': 10.5, 'bandwidth_gbps': 750}).
        """
        self._feedback_history.append(metrics)
        print(f"ProfilingFeedback: Added new metrics to history: {metrics}")

    def get_feedback_history(self) -> list:
        """
        Retrieves the complete history of performance metrics.

        Returns:
            list: A list of dictionaries, each representing a set of historical metrics.
        """
        return self._feedback_history

    def analyze_metrics(self, new_metrics: dict) -> str:
        """
        Analyzes the latest performance metrics, compares them to a baseline (if available),
        and provides specific feedback on bottlenecks for the kernel generation agent.

        Args:
            new_metrics (dict): The most recent set of performance metrics.

        Returns:
            str: A detailed summary and actionable feedback for the kernel generation agent.
        """
        feedback_lines = ["Profiling analysis and bottleneck identification:"]

        # --- Latency Analysis ---
        latency = new_metrics.get('latency_ms')
        if latency is not None:
            feedback_lines.append(f"- Current Latency: {latency:.2f} ms.")
            if self.baseline_metrics and self.baseline_metrics.get('latency_ms') is not None:
                baseline_latency = self.baseline_metrics['latency_ms']
                if latency < baseline_latency:
                    feedback_lines.append(f"  Significant improvement in latency compared to baseline ({baseline_latency:.2f} ms).")
                elif latency > baseline_latency:
                    feedback_lines.append(f"  Regression in latency compared to baseline ({baseline_latency:.2f} ms). Investigate changes.")
                else:
                    feedback_lines.append("  Latency is similar to baseline.")
            
            if self._feedback_history and len(self._feedback_history) > 1:
                prev_latency = self._feedback_history[-2].get('latency_ms', latency) # Compare to previous actual iteration
                if latency < prev_latency:
                    feedback_lines.append(f"  Latency improved by {(prev_latency - latency):.2f} ms from last iteration.")
                elif latency > prev_latency:
                    feedback_lines.append(f"  Latency regressed by {(latency - prev_latency):.2f} ms from last iteration.")

            if latency > 10.0: # General threshold for high latency
                feedback_lines.append("  Consider this a bottleneck: Current latency is high. Focus on reducing critical path, improving instruction-level parallelism, or better memory access patterns.")
            
        # --- Bandwidth Analysis ---
        bandwidth = new_metrics.get('bandwidth_gbps')
        if bandwidth is not None:
            feedback_lines.append(f"- Current HBM Bandwidth: {bandwidth:.2f} GB/s.")
            HBM_MAX_GBPS = 1228.8 # Example peak for MI300X
            utilization = (bandwidth / HBM_MAX_GBPS) * 100
            feedback_lines.append(f"  HBM Utilization: {utilization:.2f}%.")
            if utilization < 60:
                feedback_lines.append("  Consider this a bottleneck: Low HBM utilization. Optimize memory access patterns, coalesce memory accesses, increase data reuse, or use LDS effectively to reduce global memory traffic.")
            elif utilization < 80:
                feedback_lines.append("  Moderate HBM utilization. Further optimization might be possible.")
            else:
                feedback_lines.append("  Good HBM utilization.")

        # --- Occupancy Analysis ---
        occupancy = new_metrics.get('occupancy')
        if occupancy is not None:
            feedback_lines.append(f"- Current GPU Occupancy: {occupancy:.2f}.")
            if occupancy < 0.7:
                feedback_lines.append("  Consider this a bottleneck: Low occupancy. This indicates under-utilization of GPU resources (e.g., SIMDs, CUs). Explore adjusting workgroup sizes, reducing register pressure, or reducing LDS usage to allow more wavefronts/warps.")
            elif occupancy < 0.9:
                feedback_lines.append("  Moderate occupancy. Room for improvement to hide latency.")
            else:
                feedback_lines.append("  Excellent GPU occupancy.")
        
        # --- Cache Miss Rate Analysis ---
        l2_cache_miss_rate = new_metrics.get('l2_cache_miss_rate')
        if l2_cache_miss_rate is not None:
            feedback_lines.append(f"- L2 Cache Miss Rate: {l2_cache_miss_rate:.2%}.")
            if l2_cache_miss_rate > 0.05:
                feedback_lines.append("  Consider this a bottleneck: High L2 cache miss rate. Improve data locality, increase data reuse, or use prefetching techniques to reduce cache misses.")
            else:
                feedback_lines.append("  L2 cache performance appears good.")

        # --- Hardware Stalls (Simulated) ---
        # In a real scenario, rocprof would provide detailed stall reasons.
        # Here we simulate based on other metrics.
        if latency is not None and occupancy is not None and bandwidth is not None:
            if latency > 10.0 and occupancy < 0.7 and bandwidth < 700:
                feedback_lines.append("  Potential Hardware Stalls: High latency, low occupancy, and low bandwidth combined suggest significant stalls (e.g., memory stalls, instruction fetch stalls).")
            elif latency > 15.0 and l2_cache_miss_rate is not None and l2_cache_miss_rate > 0.05:
                 feedback_lines.append("  Potential Hardware Stalls: High latency with high L2 cache miss rate points to memory access stalls.")

        return "\n".join(feedback_lines)

    def get_reward_signal(self, new_metrics: dict) -> float:
        """
        Converts performance metrics into a numerical reward signal for RL-based agent training.
        This reward function aims to encourage lower latency, higher bandwidth, and higher occupancy.
        It also penalizes regressions from baseline and less optimal values.

        Args:
            new_metrics (dict): The latest performance metrics.

        Returns:
            float: A reward signal. Higher is better.
        """
        reward = 0.0

        latency = new_metrics.get('latency_ms', float('inf'))
        bandwidth = new_metrics.get('bandwidth_gbps', 0.0)
        occupancy = new_metrics.get('occupancy', 0.0)
        l2_cache_miss_rate = new_metrics.get('l2_cache_miss_rate', 1.0) # 1.0 means 100% miss rate (worst)

        # 1. Reward for Latency (primary optimization target)
        # We want to minimize latency, so inverse is good. Add a small epsilon to avoid division by zero.
        reward += (1.0 / (latency + 0.01)) * 100.0 # Scale it up

        # 2. Reward for Bandwidth Utilization
        # Assuming MI300X peak ~1228.8 GB/s
        HBM_MAX_GBPS = 1228.8
        bandwidth_util_factor = min(bandwidth / HBM_MAX_GBPS, 1.0) # Cap at 100%
        reward += bandwidth_util_factor * 50.0 # Add up to 50 points for good bandwidth

        # 3. Reward for Occupancy
        reward += occupancy * 30.0 # Add up to 30 points for good occupancy

        # 4. Penalty for L2 Cache Miss Rate
        reward -= (l2_cache_miss_rate * 100.0) * 20.0 # Penalize heavily for cache misses

        # 5. Comparison to Baseline (if available)
        if self.baseline_metrics:
            baseline_latency = self.baseline_metrics.get('latency_ms', latency)
            baseline_bandwidth = self.baseline_metrics.get('bandwidth_gbps', bandwidth)
            baseline_occupancy = self.baseline_metrics.get('occupancy', occupancy)
            baseline_l2_cache_miss_rate = self.baseline_metrics.get('l2_cache_miss_rate', l2_cache_miss_rate)

            # Reward improvement over baseline, penalize regression
            if latency < baseline_latency:
                reward += (baseline_latency - latency) * 10.0 # Reward for latency improvement
            elif latency > baseline_latency:
                reward -= (latency - baseline_latency) * 15.0 # Penalize latency regression

            if bandwidth > baseline_bandwidth:
                reward += (bandwidth - baseline_bandwidth) * 0.5 # Reward for bandwidth improvement
            elif bandwidth < baseline_bandwidth:
                reward -= (baseline_bandwidth - bandwidth) * 0.75 # Penalize bandwidth regression
            
            if occupancy > baseline_occupancy:
                reward += (occupancy - baseline_occupancy) * 10.0 # Reward for occupancy improvement
            elif occupancy < baseline_occupancy:
                reward -= (baseline_occupancy - occupancy) * 15.0 # Penalize occupancy regression

            if l2_cache_miss_rate < baseline_l2_cache_miss_rate:
                reward += (baseline_l2_cache_miss_rate - l2_cache_miss_rate) * 50.0 # Reward for lower miss rate
            elif l2_cache_miss_rate > baseline_l2_cache_miss_rate:
                reward -= (l2_cache_miss_rate - baseline_l2_cache_miss_rate) * 75.0 # Penalize higher miss rate

        # Ensure reward doesn't go too negative or infinitely high (e.g., initial states)
        return max(-1000.0, min(reward, 10000.0))
