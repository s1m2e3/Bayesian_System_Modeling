from dataclasses import dataclass
from typing import List, Union,Optional
import numpy as np
import pandas as pd


@dataclass
class OdeSimulation:
    simulation_id: int
    t: Union[List, np.ndarray]
    y: Union[List, np.ndarray]
    stochastic: bool
    def to_dataframe(self):
        self.t = self.t.reshape(len(self.t),1)
        df = pd.DataFrame(self.y)
        df['t'] = self.t
        df['simulation_id'] = self.simulation_id
        df['stochastic'] = self.stochastic
        return df


@dataclass
class IDMSimulation(OdeSimulation):
    
    t: Union[List, np.ndarray]
    y: Optional[Union[List, np.ndarray]]
    car_id:int
    velocity : Union[List, np.ndarray]
    position : Union[List, np.ndarray]
    acceleration : Optional[Union[List, np.ndarray]] = None

    def to_dataframe(self):
        df = super().to_dataframe()
        df['car_id'] = self.car_id
        df['velocity'] = self.velocity
        df['position'] = self.position
        if self.acceleration is not None:
            df['acceleration'] = self.acceleration
        return df