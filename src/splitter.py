import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.shadows import BaseShadow

@dataclass
class SplitResult:
    X_train: pd.DataFrame
    y_train: pd.Series
    
    X_test_visible: pd.DataFrame
    y_test_visible: pd.Series
    
    X_test_shadow: pd.DataFrame
    y_test_shadow: pd.Series
    
    X_test_global: pd.DataFrame
    y_test_global: pd.Series

class GapSplitter:
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

    def split(self, df: pd.DataFrame, shadow: BaseShadow) -> SplitResult:
        is_shadow = shadow.is_in_shadow(df['x'].values, df['y'].values)
        
        df_visible = df[~is_shadow].copy()
        df_shadow = df[is_shadow].copy()

        if len(df_visible) > 0:
            train_vis, test_vis = train_test_split(
                df_visible, 
                test_size=self.test_size, 
                random_state=self.random_state
            )
        else:
            train_vis = df_visible
            test_vis = df_visible

        X_train = train_vis[['x']]
        y_train = train_vis['y']

        X_test_visible = test_vis[['x']]
        y_test_visible = test_vis['y']

        X_test_shadow = df_shadow[['x']]
        y_test_shadow = df_shadow['y']
        
        X_test_global = pd.concat([X_test_visible, X_test_shadow])
        y_test_global = pd.concat([y_test_visible, y_test_shadow])

        return SplitResult(
            X_train=X_train, y_train=y_train,
            X_test_visible=X_test_visible, y_test_visible=y_test_visible,
            X_test_shadow=X_test_shadow, y_test_shadow=y_test_shadow,
            X_test_global=X_test_global, y_test_global=y_test_global
        )
