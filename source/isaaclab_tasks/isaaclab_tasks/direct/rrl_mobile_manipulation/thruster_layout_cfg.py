from typing import Dict, Tuple
from dataclasses import MISSING, field
from isaaclab.utils import configclass

@configclass
class ThrusterLayoutCfg:
    """Configuration for thruster positions and orientations."""
    
    # Thruster positions relative to body center (x, y, z)
    # Front-Right, Front-Left, Back-Right, Back-Left (pointing along X)
    # Right-Front, Left-Front, Right-Back, Left-Back (pointing along Y)
    positions: Dict[str, tuple] = field(default_factory=lambda: {
        "FR": (0.137, -0.120, 0.29757938),
        "FL": (0.137, 0.120, 0.29757938),
        "BR": (-0.137, -0.120, 0.29757938),
        "BL": (-0.137, 0.120, 0.29757938),
        "RF": (0.120, -0.137, 0.30957938),
        "LF": (0.120, 0.137, 0.30957938),
        "RB": (-0.120, -0.137, 0.30957938),
        "LB": (-0.120, 0.137, 0.30957938),
    })
    
    # Thrust directions for each thruster
    directions: Dict[str, tuple] = field(default_factory=lambda: {
        "FR": (1.0, 0.0, 0.0),   # Push forward
        "FL": (1.0, 0.0, 0.0),   # Push forward
        "BR": (-1.0, 0.0, 0.0),  # Push backward
        "BL": (-1.0, 0.0, 0.0),  # Push backward
        "RF": (0.0, -1.0, 0.0),  # Push right
        "RB": (0.0, -1.0, 0.0),  # Push right
        "LF": (0.0, 1.0, 0.0),   # Push left
        "LB": (0.0, 1.0, 0.0),   # Push left
    })
    
    # Maximum thrust force per thruster (Newtons)
    max_thrust: float = 1.0
