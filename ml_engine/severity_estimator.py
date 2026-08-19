class DynamicSeverityEstimator:
    """
    Vehicle-Only Dynamic Physics Severity Engine.
    Accident Score = Overlap Score + Velocity Change (Delta v) + Trajectory Direction Change (Delta theta)
                    + Multi-Vehicle Mass Score + Fire Bonus (25) + Smoke Bonus (10)
    """

    def __init__(self, minor_threshold=40.0, major_threshold=75.0):
        self.minor_threshold = minor_threshold
        self.major_threshold = major_threshold

        self.mass_factors = {
            'motorcycle': 0.8,
            'bike': 0.8,
            'car': 1.0,
            'bus': 1.6,
            'truck': 1.8,
            'unknown': 1.0
        }

    def estimate_severity(self, impact_force_index, velocity_delta, direction_delta=0.0, vehicle_count=1, 
                          fire_detected=False, smoke_detected=False, vehicle_types=None):
        """
        Calculate vehicle collision severity score and level.

        Returns:
        - dict: { 'severity_score': float, 'severity_level': str, 'breakdown': dict }
        """
        if vehicle_types is None:
            vehicle_types = ['car'] * max(1, vehicle_count)

        # 1. Collision Overlap / Compression Index (0 - 30 pts)
        overlap_score = min(30.0, float(impact_force_index))

        # 2. Velocity Change Delta v (0 - 30 pts)
        velocity_score = min(30.0, (float(velocity_delta) / 50.0) * 25.0)

        # 3. Direction Change Delta theta (0 - 15 pts)
        direction_score = min(15.0, (float(direction_delta) / 90.0) * 12.0)

        # 4. Multi-Vehicle Mass Multiplier (0 - 15 pts)
        mass_multiplier = max([self.mass_factors.get(str(v).lower(), 1.0) for v in vehicle_types]) if vehicle_types else 1.0
        multi_vehicle_score = min(15.0, (max(1, vehicle_count) - 1) * 7.5 * mass_multiplier)

        # 5. Post-Collision Fire & Smoke Hazard Bonuses
        fire_bonus = 25.0 if fire_detected else 0.0
        smoke_bonus = 10.0 if smoke_detected else 0.0

        # Total Weighted Physics Accident Score
        raw_score = overlap_score + velocity_score + direction_score + multi_vehicle_score + fire_bonus + smoke_bonus
        severity_score = round(min(100.0, max(0.0, raw_score)), 2)

        # Determine Severity Level
        if fire_detected and severity_score >= 50.0:
            severity_level = 'Critical'
        elif severity_score < self.minor_threshold:
            severity_level = 'Minor'
        elif severity_score < self.major_threshold:
            severity_level = 'Major'
        else:
            severity_level = 'Critical'

        return {
            'severity_score': severity_score,
            'severity_level': severity_level,
            'breakdown': {
                'overlap_score': round(overlap_score, 2),
                'velocity_score': round(velocity_score, 2),
                'direction_score': round(direction_score, 2),
                'multi_vehicle_score': round(multi_vehicle_score, 2),
                'fire_bonus': fire_bonus,
                'smoke_bonus': smoke_bonus
            }
        }
