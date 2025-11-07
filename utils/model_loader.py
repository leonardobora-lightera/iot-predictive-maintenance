"""
Model Loader with Streamlit Caching
Loads CatBoost production pipeline once and shares across all sessions
"""
import streamlit as st
import joblib
from pathlib import Path


@st.cache_resource
def load_pipeline(model_path: str = None):
    """
    Load trained CatBoost pipeline with caching
    
    Uses @st.cache_resource to load model ONCE and share across all sessions.
    This is critical for ML models (unserializable objects).
    
    Parameters
    ----------
    model_path : str, optional
        Path to .pkl file. If None, uses default production model.
    
    Returns
    -------
    pipeline : Pipeline
        Trained sklearn Pipeline (SimpleImputer → SMOTE → CatBoost)
    
    Raises
    ------
    FileNotFoundError
        If model file doesn't exist
    """
    if model_path is None:
        # Default to production model in models/ directory
        base_dir = Path(__file__).parent.parent
        model_path = base_dir / "models" / "catboost_pipeline_v1_20251107.pkl"
    
    try:
        pipeline = joblib.load(model_path)
        return pipeline
    except FileNotFoundError:
        st.error(f"❌ Model file not found: {model_path}")
        st.info("Please ensure the model file exists in models/ directory")
        raise
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        raise


@st.cache_resource
def load_metadata(metadata_path: str = None):
    """
    Load model metadata JSON with caching
    
    Parameters
    ----------
    metadata_path : str, optional
        Path to metadata JSON. If None, uses default.
    
    Returns
    -------
    metadata : dict
        Model metadata with features, hyperparameters, performance metrics
    """
    import json
    
    if metadata_path is None:
        base_dir = Path(__file__).parent.parent
        metadata_path = base_dir / "models" / "catboost_pipeline_v1_20251107_metadata.json"
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata
    except FileNotFoundError:
        st.warning(f"⚠️ Metadata file not found: {metadata_path}")
        return {}
    except Exception as e:
        st.warning(f"⚠️ Error loading metadata: {e}")
        return {}
