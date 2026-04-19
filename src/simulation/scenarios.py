


from dataclasses import dataclass

@dataclass
class AlgorithmScenario:
    """Defines the performance characteristics of an FL algorithm."""
    name: str
    target_accuracy: float
    convergence_rate: float
    midpoint: int
    noise_level: float
    
    # Radar Chart Metrics (Scale 1-10)
    privacy_score: float
    robustness_score: float
    comm_efficiency: float
    comp_efficiency: float

class ScenarioRegistry:
    """Factory for Algorithm Scenarios."""
    
    @staticmethod
    def get_scenarios():
        return {
            "Vanilla FedAvg": AlgorithmScenario(
                name="Vanilla FedAvg",
                target_accuracy=84.5,
                convergence_rate=0.06,
                midpoint=30,
                noise_level=0.5,
                privacy_score=1.0,      # No privacy
                robustness_score=4.0,   # Susceptible to poisoning
                comm_efficiency=7.0,    # Good convergence
                comp_efficiency=10.0    # Very fast
            ),
            "FedProx": AlgorithmScenario(
                name="FedProx",
                target_accuracy=86.2,
                convergence_rate=0.05,
                midpoint=35,
                noise_level=0.4,
                privacy_score=1.0,      # No privacy
                robustness_score=7.0,   # Handles heterogeneity well
                comm_efficiency=6.0,    # Slower due to proximal term optimization
                comp_efficiency=8.0
            ),
            "SCAFFOLD": AlgorithmScenario(
                name="SCAFFOLD",
                target_accuracy=87.5,
                convergence_rate=0.07,
                midpoint=28,
                noise_level=0.4,
                privacy_score=1.0,      # No privacy
                robustness_score=6.0,   # Handles drift well
                comm_efficiency=8.0,    # fast convergence
                comp_efficiency=7.0     # Control variates overhead
            ),
            "DP-FedAvg": AlgorithmScenario(
                name="DP-FedAvg",
                target_accuracy=74.0,   # Significant utility loss due to DP noise
                convergence_rate=0.04,
                midpoint=50,
                noise_level=1.5,
                privacy_score=9.0,      # High Privacy (Theoretically)
                robustness_score=5.0,
                comm_efficiency=3.0,    # Slow convergence
                comp_efficiency=9.0
            ),
            # --- New Ablation Study Scenarios ---
            "FL + Distributed Noise (No SecAgg)": AlgorithmScenario(
                name="FL + Distributed Noise",
                target_accuracy=87.0,   # Good accuracy (noise is distributed)
                convergence_rate=0.063, # Slightly slower than full
                midpoint=33,
                noise_level=0.7,
                privacy_score=5.0,      # Weak privacy without SecAgg (server sees shares)
                robustness_score=5.0,
                comm_efficiency=7.0,
                comp_efficiency=10.0    # No crypto overhead
            ),
            "FL + Secure Aggregation (No DistDP)": AlgorithmScenario(
                name="FL + Secure Aggregation",
                target_accuracy=84.5,   # Same as FedAvg (SecAgg doesn't affect acc)
                convergence_rate=0.06,
                midpoint=30,
                noise_level=0.5,
                privacy_score=6.0,      # Input privacy but no output DP
                robustness_score=8.5,
                comm_efficiency=7.0,
                comp_efficiency=7.0     # Masking overhead
            ),
            "FL + PQ Encryption Only": AlgorithmScenario(
                name="FL + PQ Encryption Only",
                target_accuracy=84.5,   # Same as FedAvg
                convergence_rate=0.06,
                midpoint=30,
                noise_level=0.5,
                privacy_score=3.0,      # Usage of encryption helps confidentiality but not DP
                robustness_score=4.0,
                comm_efficiency=6.0,    # Ciphertext expansion
                comp_efficiency=5.0     # Encryption overhead
            ),
            # ------------------------------------
            "PQ-FL (Proposed)": AlgorithmScenario(
                name="PQ-FL (Proposed)",
                target_accuracy=88.1,   # Best accuracy (Momentum + Clean Aggregation)
                convergence_rate=0.065,
                midpoint=32,
                noise_level=0.6,        # Distributed Noise impacts less than Local DP
                privacy_score=9.8,      # Post-Quantum Security + Privacy
                robustness_score=9.0,   # Secure Aggregation handles dropouts
                comm_efficiency=7.0,    # Good
                comp_efficiency=6.0     # Crypto overhead acceptable
            )
        }
